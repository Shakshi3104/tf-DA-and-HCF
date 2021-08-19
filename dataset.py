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
    print("No DA, No HCF")
    for item in ds_plain:
        print(item)

    # 2: データ拡張; .map() -> .batch()
    ds_da = tf.data.Dataset.from_tensor_slices((x, y)).map(augment).batch(5).repeat(3)
    print("DA (.map -> .batch)")
    for item in ds_da:
        print(item)

    # 3: データ拡張; .batch() -> .map()
    ds_da_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(augment_batch).repeat(3)
    print("DA (.batch -> .map")
    for item in ds_da_batch:
        print(item)

    # 4: 特徴量抽出
    ds_hcf_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(extract_batch)
    print("HCF")
    for item in ds_hcf_batch:
        print(item)

    # 5: 特徴量抽出 + 生データ
    ds_raw_hcf_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(raw_and_extract_batch)
    print("Raw and HCF")
    for item in ds_raw_hcf_batch:
        print(item)

    # 6: データ拡張 + 特徴量抽出
    ds_da_hcf_batch = tf.data.Dataset.from_tensor_slices((x, y)).batch(5).map(augment_and_extract).repeat(3)
    print("DA and HCF")
    for item in ds_da_hcf_batch:
        print(item)
