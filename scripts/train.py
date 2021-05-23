import ast
import re
from glob import glob
from os import path

import matplotlib.cm
import numpy as np
import pandas as pd
import tensorflow as tf

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()


def read_top_genres(path):
    genres = pd.read_csv(path)
    genres = genres.set_index("genre_id")
    top_titles = genres["title"][genres["top_level"].values]
    return dict(zip(genres.index, top_titles))


def read_track_data(path):
    # https://github.com/mdeff/fma/blob/master/utils.py#L197-L224
    tracks = pd.read_csv(path, index_col=0, header=[0, 1])

    COLUMNS = [
        ("track", "tags"),
        ("album", "tags"),
        ("artist", "tags"),
        ("track", "genres"),
        ("track", "genres_all"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [
        ("track", "date_created"),
        ("track", "date_recorded"),
        ("album", "date_created"),
        ("album", "date_released"),
        ("artist", "date_created"),
        ("artist", "active_year_begin"),
        ("artist", "active_year_end"),
    ]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ("small", "medium", "large")
    try:
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            "category", categories=SUBSETS, ordered=True
        )
    except (ValueError, TypeError):
        # the categories and ordered arguments were removed in pandas 0.25
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=SUBSETS, ordered=True)
        )

    COLUMNS = [
        ("track", "genre_top"),
        ("track", "license"),
        ("album", "type"),
        ("album", "information"),
        ("artist", "bio"),
    ]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype("category")

    return tracks


def spec_paths_labeled(top_genres, track_meta):
    spec_paths = glob("../data/spectrograms/*")
    spec_trkids = [int(re.search(r"([0-9]+)[.].*", f)[1]) for f in spec_paths]
    specpath_by_trkid = dict(zip(spec_trkids, spec_paths))
    labels = list(set(top_genres.values()))
    labels.sort()
    label_ids = dict(zip(labels, range(len(labels))))
    for trkid, record in track_meta.iterrows():
        if trkid not in specpath_by_trkid:
            continue
        specpath = specpath_by_trkid[trkid]
        trk_labels = [0] * len(labels)
        for genre in record["track"]["genres_all"]:
            if genre in top_genres:
                label_id = label_ids[top_genres[genre]]
                trk_labels[label_id] = 1
        yield specpath, trk_labels


@tf.function
def load_spectrogram(spec_path, labels):
    content = tf.io.read_file(spec_path)
    S = tf.io.decode_image(content, channels=3, expand_animations=False)
    return S, labels


@tf.function
def time_mask(S, h):
    dt_max = tf.cast(h * 0.2 + 0.5, tf.int32)
    t0 = tf.random.uniform((), 0, h - dt_max - 1, dtype=tf.int32)
    t1 = tf.random.uniform((), t0, t0 + dt_max, dtype=tf.int32)
    indices = tf.reshape(tf.range(h), (-1, 1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, t0),
        tf.math.less(indices, t0 + t1),
    )
    S = tf.where(condition, tf.cast(0, S.dtype), S)
    return S


@tf.function
def freq_mask(S, w):
    df_max = tf.cast(tf.cast(w, tf.float32) * 0.2 + 0.5, tf.int32)
    f0 = tf.random.uniform((), 0, w - df_max - 1, dtype=tf.int32)
    f1 = tf.random.uniform((), f0, f0 + df_max, dtype=tf.int32)
    indices = tf.reshape(tf.range(w), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f1)
    )
    S = tf.where(condition, tf.cast(0, S.dtype), S)
    return S


@tf.function
def time_dilate(S, w, h):
    max_h = tf.cast(h * 30 + 0.5, tf.int32)
    s = tf.random.uniform((), h, max_h, dtype=tf.int32)
    S = tf.image.resize(S, [s, s])
    t0 = 0
    if s > h:
        t0 = tf.random.uniform((), 0, s - h, dtype=tf.int32)
    S = tf.image.crop_to_bounding_box(S, t0, s - w, h, w)
    return S


@tf.function
def augment(S, label):
    S = tf.image.rgb_to_grayscale(S)
    S = time_dilate(S, w=224, h=224)
    S = tf.squeeze(S, axis=-1)
    S = time_mask(S, h=224)
    S = freq_mask(S, w=224)
    S = tf.expand_dims(S, axis=-1)
    S = tf.image.grayscale_to_rgb(S)
    S = tf.image.random_flip_left_right(S)
    S = tf.image.random_flip_up_down(S)
    return S, label


top_genres = read_top_genres("../data/fma_metadata/genres.csv")
track_meta = read_track_data("../data/fma_metadata/tracks.csv")
paths, labels = zip(*spec_paths_labeled(top_genres, track_meta))
paths, labels = np.array(paths, dtype=object), np.array(labels)
K_TOP_GENRES = 8
label_counts = np.sum(labels, axis=0)
top_labels = np.argsort(label_counts)[::-1]
top_labels = top_labels[:K_TOP_GENRES]
labels = labels[:, top_labels]
keep_labels = np.sum(labels, axis=1) > 0
labels = labels[keep_labels]
paths = paths[keep_labels]
labels = tf.convert_to_tensor(labels)
paths = tf.convert_to_tensor(paths)
labeled_paths = tf.data.Dataset.from_tensor_slices((paths, labels))
labeled_specs = labeled_paths.map(load_spectrogram)
train = labeled_specs.skip(1024).map(augment).prefetch(tf.data.AUTOTUNE).batch(1)
test = labeled_specs.take(1024).cache().batch(8)
# %%
persist = tf.keras.callbacks.ModelCheckpoint(
    filepath="./training/aurover.h5",
    save_weights_only=False,
    save_freq=1024,
    save_best_only=False,
    monitor="val_loss",
    verbose=0,
)
stop_early = tf.keras.callbacks.EarlyStopping(patience=1)

FINETUNING = False
cnn = tf.keras.applications.EfficientNetB0(
    weights="imagenet" if FINETUNING else None, include_top=False
)
cnn.trainable = not FINETUNING
head = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(cnn.output)
head = tf.keras.layers.BatchNormalization()(head)
head = tf.keras.layers.Dense(512, name="embed_sm")(head)
head = tf.keras.layers.Dense(K_TOP_GENRES, name="genres", activation="softmax")(head)
model = tf.keras.Model(cnn.inputs, head)
metrics = [
    tf.keras.metrics.Accuracy(name="accuracy"),
    # tf.keras.metrics.Recall(name="recall@3", top_k=3),
]
callbacks = [
    persist,
    # stop_early
]
if path.exists("./training/aurover.h5"):
    model.load_weights("./training/aurover.h5")


def double_macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    soft_f1_class0 = 2 * tn / (2 * tn + fn + fp + 1e-16)
    cost_class1 = (
        1 - soft_f1_class1
    )  # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = (
        1 - soft_f1_class0
    )  # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (
        cost_class1 + cost_class0
    )  # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    # loss=double_macro_soft_f1,
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=metrics,
)
model.fit(train, epochs=40, validation_data=test, callbacks=callbacks, verbose=1)

# %%
