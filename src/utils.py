#coding: utf-8
import pandas as pd
import os
from glob import glob
import importlib
from src import const
import argparse


parser = argparse.ArgumentParser("Center Loss Example")

# optimization
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr_cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight_cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--epoch', type=int, default=50)
# model
#parser.add_argument('--model', type=str, default='cnn')

args = parser.parse_args()



def get_train_test():
    df = pd.read_csv('data/info.csv')
    # 取出上衣
    df = df[df.clothes_type==1]
    # item_id去重
    item_id = df.item_id.drop_duplicates()
    # 前80%item_id，并转dataframe
    train_id = pd.DataFrame(item_id.iloc[:int(len(item_id) * 0.8)])
    train_df = pd.merge(df, train_id, on='item_id')
    #merge即join，默认为inner join
    # 后20%item_id，并转dataframe
    test_id = pd.DataFrame(item_id.iloc[int(len(item_id) * 0.8):])
    test_df = pd.merge(df, test_id, on='item_id')
    return train_df, test_df

