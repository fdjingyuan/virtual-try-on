#coding: utf-8
import pandas as pd
import os
from glob import glob
import importlib
from src import const
import argparse


def merge_const(module_name):
    new_conf = importlib.import_module(module_name)
    for key, value in new_conf.__dict__.items():
        if not(key.startswith('_')):
            # const.__dict__[key] = value
            setattr(const, key, value)
            print('override', key, value)


def parse_args_and_merge_const():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', default='', type=str)
    args = parser.parse_args()
    if args.conf != '':
        merge_const(args.conf)


def get_train_test():
    df = pd.read_csv('data/info.csv')
    df = df[df.clothes_type==1] # 选出上衣
    df = df.sample(frac=1)

    train_df = df.iloc[:int(len(df) * 0.9)]
    test_df = df.iloc[int(len(df) * 0.9):]
    return train_df, test_df