import numpy as np

import tensorflow as tf

from augmentation import augment_batch, augment
from features import extract_batch, raw_and_extract_batch, augment_and_extract


if __name__ == "__main__":
    # データの準備
    x = np.arange(256 * 3 * 10).reshape(-1, 256, 3)
    y = np.random.randint(0, 6, 10)
    print(x.shape, y.shape)

    # 1: データ拡張 & 特徴量抽出 なしのデータセット
    ds_plain = tf.data.Dataset.from_tensor_slices((x, y)).batch(5)

    # 2: データ拡張; .map() -> .batch()
    ds_da = tf.data.Dataset.from_tensor_slices((x, y)).map(augment).batch(5).repeat(3)

    # 3: データ拡張; .batch() -> .map()
    ds_da_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(augment_batch).repeat(3)

    # 4: 特徴量抽出
    ds_hcf_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(extract_batch)

    # 5: 特徴量抽出 + 生データ
    ds_raw_hcf_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(raw_and_extract_batch)

    # 6: データ拡張 + 特徴量抽出
    ds_da_hcf_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(augment_and_extract).repeat(3)
