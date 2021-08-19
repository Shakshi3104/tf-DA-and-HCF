import numpy as np
import scipy.stats as sp
import tensorflow as tf

from augmentation import flipping, swapping


def extracting_features(x, axis=1):
    """Extract features from 3-axis sensor data, shape=(batch, window_size, 3)
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(batch, window_size, 3)
    axis: axis
    Returns
    -------
    features
    """
    features = [
        np.max(x, axis=axis),
        np.min(x, axis=axis),
        np.mean(x, axis=axis),
        np.std(x, axis=axis),
        sp.scoreatpercentile(x, 25, axis=axis),
        np.median(x, axis=axis),
        sp.scoreatpercentile(x, 75, axis=axis),
    ]
    # x_max, y_max, z_max, x_min, ...の順になる
    return np.stack(features, 1).reshape(-1, 3 * len(features))


@tf.function
def extract_batch(x, y):
    x, y = tf.py_function(extracting_features, [x, 1], tf.float64), y
    return x, y


@tf.function
def raw_and_extract_batch(x, y):
    x, f, y = x, tf.py_function(extracting_features, [x, 1], tf.float64), y
    return (x, f), y


@tf.function
def augment_and_extract(x, y):
    x, y = tf.py_function(flipping, [x, False, 1], tf.float64), y
    x, y = tf.py_function(swapping, [x], tf.float64), y
    x, f, y = x, tf.py_function(extracting_features, [x, 1], tf.float64), y
    return (x, f), y


if __name__ == "__main__":
    x = np.arange(256 * 3 * 10).reshape(-1, 256, 3)
    y = np.random.randint(0, 6, 10)
    print(x.shape, y.shape)

    (_, feature_tensor), _ = raw_and_extract_batch(x, y)
    feature = feature_tensor.numpy()

    (augmented_tensor, feature_tensor), _ = augment_and_extract(x, y)
    augmented_x = augmented_tensor.numpy()
    augmented_feature = feature_tensor.numpy()
