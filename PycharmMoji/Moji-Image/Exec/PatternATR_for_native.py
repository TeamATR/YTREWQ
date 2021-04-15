#!/usr/bin/env python
# coding: utf-8
import random

import cv2
import math
import pickle
import numpy as np
import time
from pylsd.lsd import lsd
import os.path
from copy import deepcopy


# ファイル名生成(imgディレクトリ下)
def time_file_name(fname):
    time_path = time.strftime("%M_%H_%d_%b_", time.gmtime()) + fname
    path = os.path.join('../img', time_path)
    return path


# ランダムカラーを生成する
def generate_random_color():
    return [random.randint(0, 255) for _ in range(3)]


# 指定した座標内かどうかの判定
def arrin(x1, y1, x2, y2, array, threshold):
    a = threshold
    for row in array:
        if x1 >= row[0] - a and y1 >= row[1] - a and x1 <= row[2] + a and y1 <= row[3] + a:
            if x2 >= row[0] - a and y2 >= row[1] - a and x2 <= row[2] + a and y2 <= row[3] + a:
                return 0
    return 1


# ???
def calc_matches_knn(img1, img2, param):
    # A-KAZE検出器の生成
    detector = cv2.AKAZE_create()

    # 特徴量の検出と特徴量ベクトルの計算
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Brute-Force Matcherの生成
    bf = cv2.BFMatcher()

    # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
    matches = bf.knnMatch(des1, des2, k=2)

    # データを間引く
    ratio = param
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])

    # 特徴量をマッチング状況に応じてソート
    good = sorted(matches, key=lambda x: x[1].distance)

    # 対応する特徴点同士を描画
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:2], None, flags=2)
    # 結果保存
    cv2.imwrite(time_file_name("calc_matches_knn.png"), img3)

    # 特徴量データを取得
    q_kp = []
    t_kp = []
    for p in good[:2]:
        for px in p:
            q_kp.append(kp1[px.queryIdx])
            t_kp.append(kp2[px.trainIdx])

    # 加工対象の画像から特徴点間の角度と距離を計算
    # q_x1, q_y1 = q_kp[0]
    # q_x2, q_y2 = q_kp[-1]
    q_x1, q_y1 = q_kp[0].pt
    q_x2, q_y2 = q_kp[-1].pt

    q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
    q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)

    # テンプレート画像から特徴点間の角度と距離を計算
    # t_x1, t_y1 = t_kp[0]
    # t_x2, t_y2 = t_kp[-1]
    t_x1, t_y1 = t_kp[0].pt
    t_x2, t_y2 = t_kp[-1].pt

    t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
    t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)

    # 切出し位置の計算
    # x1 = q_x1 - t_x1 * (q_len / t_len)
    # x2 = x1 + img2.shape[1] * (q_len / t_len)
    x1 = int(q_x1 - t_x1 * (q_len / t_len))
    x2 = int(x1 + img2.shape[1] * (q_len / t_len))

    # y1 = q_y1 - t_y1 * (q_len / t_len)
    # y2 = y1 + img2.shape[0] * (q_len / t_len)
    y1 = int(q_y1 - t_y1 * (q_len / t_len))
    y2 = int(y1 + img2.shape[0] * (q_len / t_len))

    # 画像サイズ
    x, y, c = img1.shape
    size = (x, y)
    # 回転の中心位置
    center = (q_x1, q_y1)
    # 回転角度
    angle = q_deg - t_deg
    # サイズ比率
    scale = 1.0
    # 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # アフィン変換
    img_rot = cv2.warpAffine(img1, rotation_matrix, size, flags=cv2.INTER_CUBIC)
    # 画像の切出し
    img_rot = img_rot[y1:y2, x1:x2]
    # 縮尺調整
    x, y, c = img2.shape
    img_rot = cv2.resize(img_rot, (y, x))

    # 結果表示
    cv2.imwrite(time_file_name("calc_matches_knn_img_rot.png"), img_rot)


## テンプレートマッチを使ったオブジェクト検出
# def detect_object_by_matchTemplate(imgcol, img, temp):
#    result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
#    # 検出結果から検出領域の位置を取得
#    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#    top_left = max_loc
#    w, h = temp.shape[::-1]
#    bottom_right = (top_left[0] + w, top_left[1] + h)
#    # 検出領域を四角で囲んで保存
#    result = imgcol.copy()
#    cv2.rectangle(result,top_left, bottom_right, (255, 0, 0), 2)
#    cv2.imwrite(time_file_name("detect_object_matchTemplate.png"), result)


# テンプレートマッチを使ったオブジェクト検出
def detect_object_by_matchTemplate(imgcol, img, temp, param):
    arr = np.array([[2, 2, 4, 4]])
    # 比較方法はcv2.TM_CCOEFF_NORMEDを選択
    result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    # 類似度の設定(0~1)
    threshold = param
    # 検出結果から検出領域の位置を取得
    loc = np.where(result >= threshold)
    # 検出領域を四角で囲んで保存
    result2 = imgcol.copy()
    w, h = temp.shape[::-1]
    for top_left in zip(*loc[::-1]):
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(result2, top_left, bottom_right, (255, 0, 0), 2)
        arr = np.append(arr, [[top_left[0], top_left[1], top_left[0] + w, top_left[1] + h]], axis=0)
    cv2.imwrite(time_file_name("detect_object_matchTemplate.png"), result2)

    return arr


# Houghを使った線検出
def detect_line_by_HoughLinesP(imgcol, hough_param, color_param):
    # 画像をグレースケール化
    gray = cv2.cvtColor(imgcol, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(time_file_name("original_image_grayscale.png"), gray)
    gray2 = cv2.bitwise_not(gray)
    cv2.imwrite(time_file_name("original_image_grayscale_HAN.png"), gray)

    # detect
    lines = cv2.HoughLinesP(
        gray2,
        rho=hough_param["rho"],
        theta=hough_param["theta"],
        threshold=hough_param["threshold"],
        minLineLength=hough_param["minLineLength"],
        maxLineGap=hough_param["maxLineGap"]
    )

    return lines


# Canny + Houghを使った線検出
def detect_line_by_CannyHoughLinesP(img00, hough_param, color_param):
    # 画像をグレースケール化
    gray = cv2.cvtColor(img00, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 5)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # detect
    linesH = cv2.HoughLinesP(
        edges,
        rho=hough_param["rho"],
        theta=hough_param["theta"],
        threshold=hough_param["threshold"],
        minLineLength=hough_param["minLineLength"],
        maxLineGap=hough_param["maxLineGap"]
    )

    return linesH


# lsdを使った線検出
def detect_line_by_lsd(img00):
    # 画像をグレースケール化
    gray = cv2.cvtColor(img00, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 5)

    # detect
    linesL = lsd(gray)

    return linesL


# 検出した線の描画、保存
def draw_line(img, lines, color_param, file_name):
    result = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 赤線を引く
        result = cv2.line(result, (x1, y1), (x2, y2), color_param, 2)
    cv2.imwrite(time_file_name(file_name), result)


# 検出した線の描画、保存（lsd）
def draw_line_lsd(img, lines, color_param, file_name):
    # 検出した線の可視化
    img4 = img.copy()
    for line in linesL:
        x1, y1, x2, y2 = map(int, line[:4])
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 1000:
            # 赤線を引く
            img4 = cv2.line(img4, (x1, y1), (x2, y2), color_param, 2)
    cv2.imwrite(time_file_name(file_name), img4)


# 直線関連判定
def is_lines_relation(img, lines, color_param, file_name):
    # 検出した線の可視化
    img5 = img.copy()
    # ループ処理用に直線のクローン作成
    copyLines = deepcopy(lines)
    print(lines.size)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 1000:
            for copyline in copyLines:
                x1_, y1_, x2_, y2_ = map(int, copyline[:4])
                if (x2_ - x1_) ** 2 + (y2_ - y1_) ** 2 > 1000:
                    if (x2 - x1_) ** 2 + (y2 - y1_) ** 2 < 200:
                        # 線を引く
                        c = generate_random_color()
                        # print(c)
                        img5 = cv2.line(img5, (x1, y1), (x2, y2), c, 2)
                        img5 = cv2.line(img5, (x1_, y1_), (x2_, y2_), c, 2)
    cv2.imwrite(time_file_name(file_name), img5)

# 直線関連判定
def is_lines_relation2(img, lines, color_param, file_name):
    # 検出した線の可視化
    img5 = img.copy()
    # ループ処理用に直線のクローン作成
    copyLines = deepcopy(lines)
    print(lines.size)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[:4])
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 1000:
            for copyline in copyLines:
                x1_, y1_, x2_, y2_ = map(int, copyline[:4])
                if (x2_ - x1_) ** 2 + (y2_ - y1_) ** 2 > 1000:
                    if (x2 - x1_) ** 2 + (y2 - y1_) ** 2 < 200:
                        if (x2_ - x1) ** 2 + (y2_ - y1) ** 2 > 200:
                            # 線を引く
                            c = generate_random_color()
                            print(x1_, y1_, x2_, y2_)
                            print(x1, y1, x2, y2)
                            img5 = cv2.line(img5, (x1, y1), (x2, y2), c, 2)
                            img5 = cv2.line(img5, (x1_, y1_), (x2_, y2_), c, 2)
    cv2.imwrite(time_file_name(file_name), img5)


##### main part #####

if __name__ == '__main__':

    # 画像データの読み込み, グレースケール化
    img1 = cv2.imread("./res/ATRMoji00.png")
    img1_gray = cv2.imread("./res/ATRMoji00.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./res/ATRMoji01.png")
    img2_gray = cv2.imread("./res/ATRMoji01.png", cv2.IMREAD_GRAYSCALE)

    print(img1)

    # task1: knn?
    calc_matches_knn(img1, img2, 0.2)

    # task2: オブジェクト検出
    # detect_object_by_matchTemplate(img1, img1_gray, img2_gray, 1)
    detect_object_by_matchTemplate(img1, img1_gray, img2_gray, 0.8)

    # task3: 線検出
    hough_param = {"rho": 1, "theta": np.pi / 360, "threshold": 80, "minLineLength": 250, "maxLineGap": 5}
    color_param = [0, 200, 255]
    lines = detect_line_by_HoughLinesP(img1, hough_param, color_param)
    draw_line(img1, lines, color_param, "detect_line_HoughLinesP.png")

    # task4: 線検出(閾値低)
    hough_param = {"rho": 1, "theta": np.pi / 360, "threshold": 50, "minLineLength": 50, "maxLineGap": 10}
    color_param = [0, 0, 255]
    linesH = detect_line_by_CannyHoughLinesP(img1, hough_param, color_param)
    draw_line(img1, linesH, color_param, "detect_line_CannyHoughLinesP.png")

    # task5: 線検出
    color_param = [0, 0, 255]
    linesL = detect_line_by_lsd(img1)
    draw_line_lsd(img1, linesL, color_param, "detect_line_lsd.png")

    # task6: オブジェクト検出＋線検出
    arr = detect_object_by_matchTemplate(img1, img1_gray, img2_gray, 0.8)
    linesL = detect_line_by_lsd(img1)
    img2 = img1.copy()
    threshold = 10  # オブジェクト内かどうかの判定用
    for line in linesL:
        x1, y1, x2, y2 = map(int, line[:4])
        if arrin(x1, y1, x2, y2, arr, threshold):
            # 赤線を引く
            img2 = cv2.line(img2, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            pass
            # img2 = cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(time_file_name("detect_line_HoughJOUKEN-9.png"), img2)

    # task7: オブジェクト検出＋線検出
    arr = detect_object_by_matchTemplate(img1, img1_gray, img2_gray, 0.8)
    linesL = detect_line_by_lsd(img1)
    img4 = img1.copy()
    threshold = 10  # オブジェクト内かどうかの判定用
    for line in linesL:
        x1, y1, x2, y2 = map(int, line[:4])
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 > 1000:
            if arrin(x1, y1, x2, y2, arr, threshold):
                # 赤線を引く
                img4 = cv2.line(img4, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                pass
                # img4 = cv2.line(img4, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite(time_file_name("detect_line_lsd2-2.png"), img4)

    # 8
    color_param = [0, 180, 232]
    is_lines_relation2(img1, linesL, color_param, "detect_line_relaition.png")

    # 9
