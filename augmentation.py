import numpy as np
import tensorflow as tf


def _flipping(x, overall=True, axis=0):
    """
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(window_size, 3)
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


#################################################################################################
# Data augmentation per instance
def flipping(x, overall=True):
    """flipping
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(window_size, 3)
    overall: if True, flipping all axis, else flipping each axis
    Returns
    -------
    x_new: flipped x
    """
    return _flipping(x, overall, 0)


def swapping(x):
    """swapping
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(window_size, 3)
    Returns
    ----------
    x_new: swapped x
    """
    idx = np.arange(x.shape[-1]).reshape(-1, 3)
    idx = np.take_along_axis(idx, np.random.rand(*idx.shape).argsort(axis=1), axis=1).reshape(-1)

    x_new = np.zeros(x.shape)
    x_new[:, 0] = x[:, idx[0]]
    x_new[:, 1] = x[:, idx[1]]
    x_new[:, 2] = x[:, idx[2]]
    return x_new


@tf.function
def augment(x, y):
    """augment
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(window_size, 3)
    y: label data corresponding to x
    Returns
    -------
    x, y: augmented x and y
    """
    x, y = tf.py_function(flipping, [x, False], tf.float64), y
    x, y = tf.py_function(swapping, [x], tf.float64), y
    return x, y


#################################################################################################
# Data augmentation per batch
def flipping_batch(x, overall=True):
    """flipping
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(batch, window_size, 3)
    overall: if True, flipping all axis, else flipping each axis
    Returns
    -------
    x_new: flipped x
    """
    return _flipping(x, overall, 1)


def swapping_batch(x):
    """swapping
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(batch, window_size, 3)
    Returns
    ----------
    x_new: swapped x
    """
    idx = np.arange(x.shape[-1]).reshape(-1, 3)
    idx = np.take_along_axis(idx, np.random.rand(*idx.shape).argsort(axis=1), axis=1).reshape(-1)

    x_new = np.zeros(x.shape)
    x_new[:, :, 0] = x[:, :, idx[0]]
    x_new[:, :, 1] = x[:, :, idx[1]]
    x_new[:, :, 2] = x[:, :, idx[2]]
    return x_new


@tf.function
def augment_batch(x, y):
    """augment
    Parameters
    ----------
    x: channel-last 3-axis sensor data, shape=(batch, window_size, 3)
    y: label data corresponding to x
    Returns
    -------
    x, y: augmented x and y
    """
    x, y = tf.py_function(flipping_batch, [x, False], tf.float64), y
    x, y = tf.py_function(swapping_batch, [x], tf.float64), y
    return x, y
