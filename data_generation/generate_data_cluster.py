# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:48:08 2020

@author: gk
"""

#因为我们编写的程序是需要安装在系统环境路径里面的，
#所以这种绝对导入的方式是可以相对导入的，
#这个时候搜索包名的时候是在系统环境路径里面搜索，
#但是因为你的包就在这些路径的某一个路径里面，所以可以搜得到
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets
):
    """
    Generate samples from
    :param data:
    :param x_offsets:
    :param y_offsets:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes, feature_size = data.shape
    
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    df = np.load(args.traffic_df_filename)['data']
    df=np.transpose(df,(1,0,2))
    print("data shape:",df.shape)
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test_cluster = round(num_samples * 0.2)
    num_train_cluster = round(num_samples * 0.6)
    num_val_cluster = num_samples - num_test_cluster - num_train_cluster
    x_train_cluster, y_train_cluster = x[:num_train_cluster], y[:num_train_cluster]
    x_val_cluster, y_val_cluster = (
        x[num_train_cluster: num_train_cluster + num_val_cluster],
        y[num_train_cluster: num_train_cluster + num_val_cluster],
    )
    x_test_cluster, y_test_cluster = x[-num_test_cluster:], y[-num_test_cluster:]

    for cat in ["train_cluster", "val_cluster", "test_cluster"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat] #"x"+_train/val/test,locals 所有变量字典形式，所以可以搜索
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz"%cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/XiAn_City", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="data/xian/original_data_cluster.npz", help="Raw traffic readings.",)
    parser.add_argument("--seq_length_x", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )
    parser.add_argument("--dow", action='store_true',)

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input('%s exists. Do you want to overwrite it? (y/n)'%args.output_dir)).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)

