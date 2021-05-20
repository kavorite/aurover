import os
import re
import sys
from concurrent import futures
from glob import glob

import av
import matplotlib.cm
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.ops.gen_batch_ops import batch
from tqdm import tqdm

DEBUG = getattr(sys, "gettrace", lambda: None)() is None
tf.config.run_functions_eagerly(DEBUG)


@tf.function
def log_mel_spectrogram(
    y,
    sr,
    width,
    height,
    colormap=None,
    freq_mask_factor=1 / 8,
    time_mask_factor=1 / 8,
    training=None,
):
    """
    log_mel_spectrogram is a function that generates spectrograms from audio
    signals represented by a tensor, then uses logarithmic and
    mel scaling to map the visualization of the given signal to the
    frequency response of the human ear.

    parameters
    ----------
    size: dimension of spectrogram to return
    rate: sample-rate of input signals
    colormap:
        matplotlib.cm colormap name, e.g. "viridis" (requires
        matplotlib), a 256x3 tensor mapping discretized intensities to
        RGB-encoded colors, or None (grayscale).
    freq_mask_factor:
        Sparse, real value giving the maximum proportion of contiguous
        frequencies to be masked for data augmentation during training.
    time_mask_factor:
        Sparse, real value giving the maximum proportion of contiguous
        timesteps to be masked for data augmentation during training.
    training:
        Useful for deploying this function as a lambda layer. Flag specifying
        whether time/frequency masking should be enabled. By default, the TF
        Keras backend is queried for this information.

    references
    ----------
    https://www.tensorflow.org/io/tutorials/audio
    https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/core.py#L219-L221
    """
    if not (0 < time_mask_factor < 1.0):
        raise ValueError(
            "You cannot mask under 0% or over 100% of a training instance."
        )
    if not (0 < freq_mask_factor < 1.0):
        raise ValueError(
            "You cannot mask under 0% or over 100% of a training instance."
        )
    if isinstance(colormap, str):
        colormap = tf.constant(matplotlib.cm.get_cmap(colormap).colors)
    if training is None:
        training = tf.keras.backend.learning_phase()
    S = tfio.experimental.audio.spectrogram(
        y,
        nfft=height * 2 - 1,
        window=height * 2 - 1,
        stride=y.shape[0] // (height - 1),
    )
    S = tf.math.log(S + 1e-6)
    S = tfio.experimental.audio.melscale(S, rate=sr, mels=width, fmin=0, fmax=sr // 2)
    if training:
        fmask = int(freq_mask_factor * width)
        tmask = int(time_mask_factor * height)
        S = tfio.experimental.audio.freq_mask(S, param=fmask)
        S = tfio.experimental.audio.time_mask(S, param=tmask)
    s_min = tf.reduce_min(S)
    s_max = tf.reduce_max(S)
    S = (S - s_min) / (s_max - s_min)
    if colormap is not None:
        S = tf.cast(tf.round(S * 255), tf.int32)
        S = tf.gather(colormap, S)
    return S


def decode_as_mono(container: av.container.InputContainer):
    audio = container.streams.audio[0]
    frames = container.decode(audio)
    y = np.hstack([frame.to_ndarray().mean(axis=0) for frame in frames])
    sr = audio.rate
    return y, sr


def dump_spec_tiles(opath, mp3):
    if os.path.exists(opath + ".png"):
        return
    try:
        with av.open(mp3) as container:
            y, sr = decode_as_mono(container)
    except (IndexError, ValueError):
        return
    S = log_mel_spectrogram(y, sr, width=224, height=224, colormap=None)
    S = tf.reshape(S, [224, 224, 1])
    S = tf.image.grayscale_to_rgb(S)
    S = tf.cast(S * (1 << 16), tf.uint16)
    dirname = os.path.dirname(opath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    tf.io.write_file(opath + ".png", tf.io.encode_png(S))


def sanitize_metadata(raw: str) -> str:
    return "_".join(re.split(r"[^\w]+", raw.lower()))


def load_genre(mp3):
    try:
        with av.open(mp3) as container:
            raw = container.metadata["genre"]
            return sanitize_metadata(raw)
    except:
        return


def main():
    output_dir = "../data/spectrograms"
    av.logging.set_level(av.logging.FATAL)

    mp3s = np.array(glob("../data/fma_medium/**/*"), dtype=object)
    # genre_ledger = pd.read_csv("../data/fma_metadata/genres.csv")
    # genre_ledger["title"] = genre_ledger["title"].map(sanitize_metadata)
    # genre_ledger = genre_ledger.set_index("genre_id")
    # genre_ledger = genre_ledger.join(
    #     genre_ledger,
    #     on="top_level",
    #     how="inner",
    #     rsuffix="_top",
    # )
    # map each genre to its top-level entry
    # top_level = dict(zip(genre_ledger["title"], genre_ledger["title_top"]))
    # print("Load genres...")
    # with futures.ThreadPoolExecutor(6) as pool:
    #     genres = tqdm(pool.map(load_genre, mp3s), total=len(mp3s))
    #     genres = [top_level.get(g, None) for g in genres]
    # genres = np.array(genres, dtype=object)
    # mp3s = mp3s[genres != None]
    # genres = genres[genres != None]
    # opaths = np.array(
    #     [
    #         os.path.join(output_dir, genre, os.path.basename(mp3))
    #         for genre, mp3 in zip(genres, mp3s)
    #     ],
    #     dtype=object,
    # )
    opaths = [os.path.join(output_dir, os.path.basename(mp3)) for mp3 in mp3s]
    print("Generate spectrogram tiles...")
    with futures.ThreadPoolExecutor(6) as pool:
        for _ in tqdm(
            pool.map(lambda args: dump_spec_tiles(*args), zip(opaths, mp3s)),
            total=len(mp3s),
        ):
            pass


if __name__ == "__main__":
    main()
