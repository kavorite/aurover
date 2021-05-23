import ctypes
from os.path import abspath, dirname

import aubio
import av
import numpy as np
import pythoncom
import tensorflow as tf
import win32clipboard
from resampy import resample


def detect_bpm(y, sr, beat_detector=None):
    y = np.array(y).astype(np.float32)
    if beat_detector is None:
        beat_detector = aubio.tempo(method="specdiff", samplerate=int(sr))
    buf_len = beat_detector.hop_size
    beats = []
    while len(y) > buf_len:
        samples = y[:buf_len]
        is_beat = beat_detector(samples).astype(np.float32)
        if is_beat:
            beats.append(beat_detector.get_last_s())
        y = y[buf_len + 1 :]
    bpms = 60 / np.diff(beats)
    return np.median(bpms)


def decode_k_samples(container, audio, k, silence=1e-4):
    n = 0
    for frame in container.decode(audio):
        yy = frame.to_ndarray().mean(axis=0)
        yy = yy[yy > silence]
        n += len(yy)
        yield yy
        if n >= k:
            break


def partial_decode(container, duration, num_segments=3, silence=1e-4):
    audio = container.streams.audio[0]
    sr = audio.rate
    tot_samples = int(duration * sr + 0.5)
    segments = []
    segment_len = int(tot_samples / num_segments)
    cursor = 0
    while cursor < container.duration and len(segments) < num_segments:
        container.seek(cursor)
        segment = list(decode_k_samples(container, audio, segment_len, silence))
        if segment:
            segments.append(np.hstack(segment))
        cursor += container.duration // num_segments
        cursor = max(cursor - segment_len, 0)
    y = np.zeros(sum(map(len, segments)))
    y_max = -np.inf
    y_min = np.inf
    for i, segment in enumerate(segments):
        if i >= num_segments - 1:
            break
        if len(segment) > 0:
            y_max = max(y_max, segment.max())
            y_min = min(y_min, segment.min())
        lpad = i * segment_len
        rpad = max(0, len(y) - len(segment) - lpad)
        segment = np.pad(segment, (lpad, rpad))
        segment /= num_segments
        y += segment
    target_sr = int(sr / (len(y) / tot_samples) + 0.5)
    y = resample(y, sr, target_sr)[:tot_samples]
    y -= y.min()
    y /= y.max()
    y *= y_max
    return y, sr


@tf.function
def log_mel_spectrogram(
    y,
    sr,
    width,
    height,
):
    S = tf.signal.stft(
        y,
        frame_length=height * 2 - 1,
        frame_step=y.shape[0] // (height - 1),
        fft_length=height * 2 - 1,
        window_fn=tf.signal.hann_window,
        pad_end=True,
    )
    S = tf.math.abs(S)
    S = tf.math.log(S + 1e-16)
    # mel-scale
    matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=width,
        num_spectrogram_bins=tf.shape(S)[-1],
        sample_rate=sr,
        lower_edge_hertz=0,
        upper_edge_hertz=sr // 2,
    )
    S = tf.cast(S, tf.float32)
    S = S @ matrix
    s_min = tf.reduce_min(S)
    s_max = tf.reduce_max(S)
    S = (S - s_min) / (s_max - s_min)
    return S


def standardize(x, axis=None):
    x_min = tf.reduce_min(x, axis)
    x_max = tf.reduce_max(x, axis)
    return (x - x_min) / (x_max - x_min)


def pairwise_cos_distance(points):
    points = tf.linalg.l2_normalize(points, axis=-1)
    return 1 - points @ tf.transpose(points)


def pagerank(A, alpha=0.85):
    A = alpha * A + (1 - alpha) / A.shape[1]
    eigvals, eigvecs = np.linalg.eigh(A)
    r = eigvecs[np.argmax(eigvals)]
    r -= r.min()
    if True in (x != 0 for x in r):
        r /= r.max()
    return r


def clip_files(files):
    class DROPFILES(ctypes.Structure):
        _fields_ = (
            ("pFiles", ctypes.wintypes.DWORD),
            ("pt", ctypes.wintypes.POINT),
            ("fNC", ctypes.wintypes.BOOL),
            ("fWide", ctypes.wintypes.BOOL),
        )

    if not len(files):
        return

    offset = ctypes.sizeof(DROPFILES)
    length = sum(len(p) + 1 for p in files) + 1
    size = offset + length * ctypes.sizeof(ctypes.c_wchar)
    buf = (ctypes.c_char * size)()
    df = DROPFILES.from_buffer(buf)
    df.pFiles, df.fWide = offset, True
    for path in files:
        array_t = ctypes.c_wchar * (len(path) + 1)
        path_buf = array_t.from_buffer(buf, offset)
        path_buf.value = path
        offset += ctypes.sizeof(path_buf)
    stg = pythoncom.STGMEDIUM()
    stg.set(pythoncom.TYMED_HGLOBAL, buf)
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    try:
        win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, stg.data)
    finally:
        win32clipboard.CloseClipboard()


def main():
    import sys
    from argparse import ArgumentParser
    from concurrent import futures
    from glob import glob

    from python_tsp.heuristics import solve_tsp_simulated_annealing
    from sklearn import cluster
    from tqdm import tqdm

    sys.stdout.reconfigure(encoding="utf8")
    parser = ArgumentParser()

    parser.add_argument(
        "--tracks", action="extend", type=splat_pattern, nargs="+", required=True
    )

    def splat_pattern(pattern):
        return glob(pattern, recursive=True)

    def positive_real(x):
        x = float(x)
        if x <= 0:
            raise ValueError("This argument must be positive")

    def song_path(path):
        try:
            with av.open(path) as container:
                rate = container.streams.audio[0].rate
            return abspath(path)
        except:
            raise ValueError(f"{path} does not appear to be a multimedia container")

    def sparse_real(x):
        x = float(x)
        if not 0 < x < 1.0:
            raise ValueError("Scale factors must lie on the interval [0, 1]")

    def natural_int(x):
        x = int(x)
        if x < 0:
            raise ValueError("natural numbers must be positive or zero")

    parser.add_argument(
        "--read-duration",
        type=positive_real,
        default=6.0,
        help="duration of each track to process in seconds",
    )
    parser.add_argument(
        "--model",
        default=dirname(abspath(__file__)) + "/aurover.h5",
        type=tf.keras.models.load_model,
    )
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--centroid", type=song_path, action="extend", nargs="*")
    parser.add_argument("--reduce", action="store_true", default=False)
    parser.add_argument("--dump", action="store_true")
    parser.add_argument("--constellations", default=None, type=natural_int)
    parser.add_argument("--max-tempo-penalty", default=1.0, type=sparse_real)
    args = parser.parse_args()
    tracks = []
    for tracklist in args.tracks:
        tracks.extend(map(abspath, tracklist))
    tracks = np.array(tracks, dtype=object)
    sys.stderr.write("loading audio files...\n")
    with futures.ThreadPoolExecutor(6) as pool:
        image_shape = args.model.inputs[0].shape[1:3]

        def decode_one(track):
            try:
                with av.open(track) as container:
                    try:
                        y, sr = partial_decode(container, args.read_duration)
                        return y, sr
                    except IndexError:
                        sys.stderr.write(
                            f"{track} does not appear to have any readable audio channels; skipping...\n"
                        )
            except av.FFmpegError:
                sys.stderr.write(
                    f"{track} does not appear to be a multimedia container; skipping...\n"
                )
                return None, None

        ys, srs = zip(*tqdm(pool.map(decode_one, tracks), total=len(tracks)))
        ys = [y for y in ys if y is not None]
        srs = [sr for sr in srs if sr is not None]

    embedder = tf.keras.Model(
        args.model.inputs, args.model.get_layer("embed_sm").output
    )

    @tf.function
    def preprocess(inputs):
        y, sr = inputs
        x = log_mel_spectrogram(y, sr, *image_shape)
        x = tf.expand_dims(x, axis=-1)
        x = tf.image.grayscale_to_rgb(x)
        return x

    tf.config.run_functions_eagerly(True)

    pad_to = max(map(len, ys))
    ys = [np.pad(y, (0, pad_to - len(y))) for y in ys]
    ys = np.vstack(ys)
    sys.stderr.write("encoding spectrograms...\n")
    spectrograms = tf.map_fn(
        preprocess,
        (tf.stack(ys), tf.cast(tf.stack(srs), tf.float32)),
        fn_output_signature=(tf.TensorSpec(shape=(*image_shape, 3))),
    )
    sys.stderr.write("extracting dense features...\n")
    embeddings = embedder.predict(spectrograms, batch_size=8)
    # A_min = tf.reduce_min(A)
    # A = tf.math.divide_no_nan(A - A_min, tf.reduce_max(A) - A_min)
    bpms = np.array([detect_bpm(y, sr) for y, sr in zip(ys, srs)], dtype=np.float32)
    bpm_distance = 1 - standardize(bpms[:, None] - bpms[None, :])
    bpm_distance = tf.clip_by_value(bpm_distance, args.max_tempo_penalty, 1.0)
    bpm_distance = tf.cast(bpm_distance, tf.float32)
    cos_distance = pairwise_cos_distance(embeddings)
    A = np.array(standardize(bpm_distance * cos_distance))
    r = pagerank(A)
    track_centrality = np.argsort(r)[::-1]
    if args.dump:
        if sys.stdout.isatty():
            print(A)
        else:
            np.save(sys.stdout.buffer, A, allow_pickle=False)
            sys.stdout.flush()
        return

    if args.constellations:
        import json

        constellations = args.constellations
        if constellations < 0:
            constellations = int(len(tracks) / 8)
        labels = cluster.KMeans(n_clusters=constellations).fit_predict(
            embeddings * r[:, None]
        )
        num_labels = max(labels) + 1
        network = [[] for _ in range(num_labels)]
        cluster_centrality = np.zeros((num_labels,))
        for i, label in enumerate(labels):
            network[label].append(i)
            cluster_centrality[label] += r[i]
        cluster_centrality -= cluster_centrality.min()
        cluster_centrality /= cluster_centrality.sum()
        network = [np.array(subnet) for subnet in network]
        manifest = [
            dict(
                centrality=centrality.item(),
                tracks=list(
                    dict(centrality=centrality, name=name)
                    for centrality, name in zip(
                        r.astype(float)[constellation], tracks[constellation]
                    )
                ),
            )
            for constellation, centrality in zip(network, cluster_centrality)
        ]
        json.dump(manifest, sys.stdout)
        return

    sys.stderr.write("computing the path...\n")
    if args.centroid:
        centroids = embeddings[np.isin(tracks, args.centroid)]
        if args.reduce:
            centroids = tf.expand_dims(tf.reduce_mean(centroids, axis=0), axis=0)
        order_scores = embeddings * centroids[:, None]
        order_scores = tf.reduce_prod(tf.linalg.norm(order_scores, axis=-1), axis=0)
        track_order = tf.argsort(order_scores)
        track_order = track_order.numpy()
    else:
        track_order = track_centrality

    track_order = track_order[: args.top_k]
    interleaved = np.zeros_like(track_order, dtype=int)
    from_front = track_order[: len(track_order) // 2]
    from_back = track_order[::-1][: len(track_order) - len(from_front)]
    interleaved[0::2] = from_front
    interleaved[1::2] = from_back
    A = A[interleaved][:, interleaved]
    tour, total_cost = solve_tsp_simulated_annealing(A)
    tracks = tracks[track_order[tour]]
    # clip_files(tracks[np.array(tour)])
    sys.stdout.write("\n".join(tracks) + "\n")
    sys.stderr.write("copying paths to clipboard...\n")
    clip_files(tracks)
    # if sys.stdout.isatty():
    # sys.stdout.write("\n".join(tracks) + "\n")
    # else:
    #     sys.stderr.write("serializing the archive...\n")
    #     basenames = [os.path.basename(track) for track in tracks]
    #     elements = [xspf.Track("file:///" + name for name in basenames)]
    #     playlist = xspf.Playlist(trackList=elements)
    #     with ZipFile(sys.stdout.buffer, "w") as archive:
    #         for track, name in zip(tracks, tqdm(basenames)):
    #             with archive.open(name, "w") as ostrm:
    #                 with open(track, "rb") as istrm:
    #                     ostrm.write(istrm.read())
    #         with archive.open("playlist.xspf", "w") as ostrm:
    #             ostrm.write(playlist.xml_string().encode("utf8"))


if __name__ == "__main__":
    main()
