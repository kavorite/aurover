import csv
import os
import re
from collections import defaultdict
from glob import glob
from io import IOBase
from multiprocessing import Pool
from typing import Union

import av
import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import istarmap


def standardize(A: np.ndarray) -> np.ndarray:
    mu = A.mean()
    R = A - mu
    sd = np.linalg.norm(R, 2)
    Mstd = R / (sd + 1e-6)
    nmin, nmax = Mstd.min(), Mstd.max()
    if nmax - nmin > 1e-6:
        S = Mstd
        S[S < nmin] = nmin
        S[S > nmax] = nmax
        S = (S - nmin) / (nmax - nmin)
    else:
        S = np.zeros_like(A)
    return S


def decode_as_mono(container: av.container.InputContainer):
    audio = container.decode(audio=0)
    frames = list(audio)
    sr = frames[0].sample_rate
    y = np.hstack([frame.to_ndarray().mean(axis=0) for frame in frames])
    return y, sr


def sample_strides(samples: np.ndarray, sample_rate: int, strides: int = 10):
    stride_length = len(samples) // strides
    return [
        samples[i * stride_length : (i + 1) * stride_length] for i in range(strides)
    ]


def spectrogram(y, sr):
    M = librosa.feature.melspectrogram(y, sr)
    M = np.array(M)
    # get rid of undefined domain for logarithms
    M[M == 0] = 1e-6
    M = standardize(np.flip(np.log(M)))
    return M


def dump_spec_tiles(opath, mp3):
    amp = spectrogram(*decode_as_mono(av.open(mp3)))
    amp = amp.dot(255).astype(np.uint8)
    # separate into horizontal segments of the width used by EfficientNetB0
    width = amp.shape[1]
    segmentc, remainder = np.divmod(width, 224)
    width -= remainder
    amp = amp[:, :width]
    gray_tiles = np.split(amp, segmentc, axis=1)
    name, ext = os.path.splitext(opath)
    tile_names = [f"{name}_tile{i}{ext}.png" for i in range(1, segmentc + 1)]
    for pair in zip(tile_names, gray_tiles):
        dump_tile(*pair)


def dump_tile(tile_name, tile):
    if not os.path.exists(os.path.dirname(tile_name)):
        os.makedirs(os.path.dirname(tile_name))
    if len(tile.shape) < 3:
        tile = cv2.applyColorMap(tile, cv2.COLORMAP_VIRIDIS)
        tile = cv2.resize(tile, (224, 224))
    ret, buf = cv2.imencode(".png", tile)
    with open(tile_name, "wb+") as ostrm:
        ostrm.write(buf)


def sanitize_metadata(raw: str) -> str:
    return "_".join(re.split(r"[^\w]+", raw.lower()))


def load_genre(mp3):
    try:
        container = av.open(mp3)
    except:
        return

    if "genre" not in container.metadata:
        return
    raw = container.metadata["genre"]
    return sanitize_metadata(raw)


def main():
    output_dir = "./data/spectrograms"
    av.logging.set_level(av.logging.FATAL)

    mp3s = np.array(glob("./data/fma_small/**/*.mp3"), dtype=object)
    genre_ledger = pd.read_csv("./data/fma_metadata/genres.csv")
    genre_ledger["title"] = genre_ledger["title"].map(sanitize_metadata)
    genre_ledger = genre_ledger.merge(
        genre_ledger,
        left_on="top_level",
        right_on="genre_id",
        how="inner",
        suffixes=("", "_top"),
    )
    # map each genre to its top-level entry
    top_level = dict(zip(genre_ledger["title"], genre_ledger["title_top"]))
    print("Load genres...")
    with Pool(cpu_count()) as pool:
        genres = tqdm(pool.imap(load_genre, mp3s), total=len(mp3s))
        genres = np.array(list(genres), dtype=object)
    genres = [top_level.get(title, None) for title in genres]
    genres = np.array(genres, dtype=object)
    with_genre = np.array([title is not None for title in genres])
    # filter out entries without genres
    genres = genres[with_genre]
    mp3s = mp3s[with_genre]
    opaths = np.array(
        [
            os.path.join(output_dir, genre, os.path.basename(mp3))
            for genre, mp3 in zip(genres, mp3s)
        ],
        dtype=object,
    )
    print("Generate spectrogram tiles...")
    with Pool() as pool:
        list(tqdm(pool.istarmap(dump_spec_tiles, zip(opaths, mp3s)), total=len(mp3s)))


if __name__ == "__main__":
    main()
