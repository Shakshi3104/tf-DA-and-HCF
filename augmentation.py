import numpy as np
import tensorflow as tf


def flipping(x, overall=True, axis=0):
    """
    Parameters
    ----------
    x: channel-last sensor data, shape=(window_size, channel) or (batch_size, window_size, channel)
    overall: if True, flipping all axis, else flipping each axis
    axis: axis
    Returns
    -------
    x_new: flipped x
    """
    if overall:
        x_new = ((-1) ** np.random.choice([0, 1])) * x
    else:
        def flip(x):
            return x * (-1) ** np.random.choice([0, 1])

        x_new = np.apply_along_axis(flip, axis, x)
    return x_new


def swapping(x):
    """swapping
    Parameters
    ----------
    x: channel-last 3-axis or 6-axis sensor data, shape=(window_size, channel) or (batch_size, window_size, channel)
    Returns
    ----------
    x_new: swapped x
    """
    # 3-axis
    if x.shape[-1] == 3:
        idx = np.arange(x.shape[-1]).reshape(-1, 3)
        idx = np.take_along_axis(idx, np.random.rand(*idx.shape).argsort(axis=1), axis=1).reshape(-1)

        x_new = np.zeros(x.shape)
        x_new[..., 0] = x[..., idx[0]]
        x_new[..., 1] = x[..., idx[1]]
        x_new[..., 2] = x[..., idx[2]]

    # 6-axis
    elif x.shape[-1] == 6:
        idx_acc = np.arange(3).reshape(-1, 3)
        idx_acc = np.take_along_axis(idx_acc, np.random.rand(*idx_acc.shape).argsort(axis=1), axis=1).reshape(-1)

        idx_gyr = np.arange(3, 6).reshape(-1, 3)
        idx_gyr = np.take_along_axis(idx_gyr, np.random.rand(*idx_gyr.shape).argsort(axis=1), axis=1).reshape(-1)

        x_new = np.zeros(x.shape)
        x_new[..., 0] = x[..., idx_acc[0]]
        x_new[..., 1] = x[..., idx_acc[1]]
        x_new[..., 2] = x[..., idx_acc[2]]

        x_new[..., 3] = x[..., idx_gyr[0]]
        x_new[..., 4] = x[..., idx_gyr[1]]
        x_new[..., 5] = x[..., idx_gyr[2]]

    return x_new


# Data augmentation per instance
@tf.function
def augment(x, y):
    """augment
    Parameters
    ----------
    x: channel-last sensor data, shape=(window_size, channel)
    y: label data corresponding to x
    Returns
    -------
    x, y: augmented x and y
    """
    x, y = tf.py_function(flipping, [x, False, 0], tf.float64), y
    x, y = tf.py_function(swapping, [x], tf.float64), y
    return x, y


# Data augmentation per batch
@tf.function
def augment_batch(x, y):
    """augment
    Parameters
    ----------
    x: channel-last sensor data, shape=(batch_size, window_size, channel)
    y: label data corresponding to x
    Returns
    -------
    x, y: augmented x and y
    """
    x, y = tf.py_function(flipping, [x, False, 1], tf.float64), y
    x, y = tf.py_function(swapping, [x], tf.float64), y
    return x, y