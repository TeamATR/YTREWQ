#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# usage: ./increase_picture.py hogehoge.jpg
#

import cv2
import numpy as np
import sys
import os


# ヒストグラム均一化
def equalizeHistRGB(src):
    RGB = cv2.split(src)
    Blue = RGB[0]
    Green = RGB[1]
    Red = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])

    img_hist = cv2.merge([RGB[0], RGB[1], RGB[2]])
    return img_hist


def routate(src,r,k):
    # 高さを定義
    height = src.shape[0]
    # 幅を定義
    width = src.shape[1]
    # 回転の中心を指定
    center = (int(width / 2), int(height / 2))
    # 回転角を指定
    angle = r
    # スケールを指定
    scale = k
    # getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    # アフィン変換
    image2 = cv2.warpAffine(src, trans, (width, height), borderValue=(255, 255, 255))
    return image2



if __name__ == '__main__':
    # ルックアップテーブルの生成
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    gamma1 = 0.75
    gamma2 = 1.5

    LUT_HC = np.arange(256, dtype='uint8')
    LUT_LC = np.arange(256, dtype='uint8')
    LUT_G1 = np.arange(256, dtype='uint8')
    LUT_G2 = np.arange(256, dtype='uint8')

    LUTs = []

    # 平滑化用
    average_square = (10, 10)

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0

    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table

    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # その他LUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

    LUTs.append(LUT_HC)
    LUTs.append(LUT_LC)
    LUTs.append(LUT_G1)
    LUTs.append(LUT_G2)

    # 画像の読み込み
    fname = "./bulb2.png"

    img_src = cv2.imread(fname, 1)
    trans_img = []
    trans_img.append(img_src)


    # 回転
    for r in range(30, 360, 20):
        scale = np.random.rand()*0.2 + 0.8
        img22 = routate(img_src,r,scale)
        trans_img.append(img22)


    # 保存
    if not os.path.exists("PRE_trans_images"):
        os.mkdir("PRE_trans_images")

    base = os.path.splitext(os.path.basename(fname))[0] + "_R_"
    img_src.astype(np.float64)
    for i, img in enumerate(trans_img):
        # 比較用
        # cv2.imwrite("PRE_trans_images/" + base + str(i) + ".jpg" ,cv2.hconcat([img_src.astype(np.float64), img.astype(np.float64)]))
        cv2.imwrite("./PRE_trans_images/" + base + str(i) + ".jpg", img)





