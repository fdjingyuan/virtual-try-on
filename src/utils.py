#coding: utf-8
import pandas as pd
import os
from glob import glob
import importlib
from src import const




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

