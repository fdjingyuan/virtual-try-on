# coding: utf-8
import pandas as pd
import os
import re

base_path = '/home/jyliu/dataset/in-shop-Clothes-Retrieval/'

## 处理item，并转换为id

df = pd.read_csv(base_path + 'Anno/list_color_cloth.txt', sep=r'\s{2,}', skiprows=1, engine='python')
def parse_id(x):
    line = x.split('/')
    filename = line[-1]
    return line[-2] + '#' + filename.split('_')[0]  # 变成 "id_00000001#02"这种形式
df['item_name'] = df.image_name.map(parse_id)
item2id = {name: i for i, name in enumerate(df.item_name.drop_duplicates())}
df['item_id'] = df['item_name'].map(lambda x: item2id[x])
df.set_index('image_name', inplace=True)

## Landmark、衣服类型的处理

with open(base_path + 'Anno/list_landmarks_inshop.txt') as f:
    f.readline()
    f.readline()
    values = []
    for line in f:
        info = re.split('\s+', line)
        image_name = info[0].strip()
        clothes_type = int(info[1])
        variation_type = int(info[2])
        # 1: ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]
        # 2: ["left waistline", "right waistline", "left hem", "right hem"]
        # 3: ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].

        landmark_postions = [(0, 0)] * 8
        landmark_visibilities = [1] * 8  # 1表示可见，0表示被遮挡或者根本不在图里（原来有三种状态）
        landmark_in_pic = [1] * 8  # 1在图里（可见或被遮挡），0表示不在图里
        landmark_info = info[3:]
        if clothes_type == 1:  # 上身
            # 从上半身到全身的index映射
            convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 6, 5: 7}
        elif clothes_type == 2:
            convert = {0: 4, 1: 5, 2: 6, 3: 7}
        elif clothes_type == 3:
            convert = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
        for i in convert:
            x = int(landmark_info[i * 3 + 1])
            y = int(landmark_info[i * 3 + 2])
            vis = int(landmark_info[i * 3])
            if vis == 2:
                in_pic = 0  # 没有的
            elif vis == 1:
                in_pic = 1  # 有遮挡的
            else:
                in_pic = 1  # 1表示露出，和属性对齐
            if vis == 2:
                vis = 0  # 没有的
            elif vis == 1:
                vis = 0  # 有遮挡的
            else:
                vis = 1  # 1表示露出，和属性对齐
            landmark_postions[convert[i]] = (x, y)
            landmark_visibilities[convert[i]] = vis
            landmark_in_pic[convert[i]] = in_pic
        tmp = []
        for pair in landmark_postions:
            tmp.append(pair[0])
            tmp.append(pair[1])
        landmark_postions = tmp  # 展平

        line_value = []
        line_value.extend([image_name, clothes_type, variation_type])
        line_value.extend(landmark_postions)
        line_value.extend(landmark_visibilities)
        line_value.extend(landmark_in_pic)
        values.append(line_value)

name = ['image_name', 'clothes_type', 'variation_type']
name.extend(['lm_lc_x', 'lm_lc_y', 'lm_rc_x', 'lm_rc_y',
             'lm_ls_x', 'lm_ls_y', 'lm_rs_x', 'lm_rs_y',
             'lm_lw_x', 'lm_lw_y', 'lm_rw_x', 'lm_rw_y',
             'lm_lh_x', 'lm_lh_y', 'lm_rh_x', 'lm_rh_y'])

name.extend([
    'lm_lc_vis', 'lm_rc_vis',
    'lm_ls_vis', 'lm_rs_vis',
    'lm_lw_vis', 'lm_rw_vis',
    'lm_lh_vis', 'lm_rh_vis',
])

name.extend([
    'lm_lc_in_pic', 'lm_rc_in_pic',
    'lm_ls_in_pic', 'lm_rs_in_pic',
    'lm_lw_in_pic', 'lm_rw_in_pic',
    'lm_lh_in_pic', 'lm_rh_in_pic',
])

landmarks = pd.DataFrame(values, columns=name)
landmarks.set_index('image_name', inplace=True)

## 合并信息

df = df.join(landmarks)
df.reset_index('image_name', inplace=True)

df.image_name = df.image_name.map(lambda x: base_path + x)

if os.path.exists('data') is False:
    os.makedirs('data')
df.to_csv('data/info.csv', index=False)