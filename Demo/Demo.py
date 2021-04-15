#!/usr/bin/env python
# coding: utf-8

# for common functions
import math
import pickle
import numpy as np
import time
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# for handling image
import cv2
from PIL import Image

# for detecting line
from pylsd.lsd import lsd

# for pyocr
import pyocr
import pyocr.builders

# ファイル名生成
def time_file_name(fname):
    return time.strftime("%d_%b_%Y_", time.gmtime()) + fname


# 指定した座標内かどうかの判定
def arrin(x1, y1, x2, y2, array, threshold):
    a = threshold
    for row in array:
        # 座標は左上原点系
        # top_leftとの比較
        if x1>=row[0]-a and y1>=row[1]-a and x1<=row[2]+a and y1<=row[3]+a:
            # bottom_rightとの比較
            if x2>=row[0]-a and y2>=row[1]-a and x2<=row[2]+a and y2<=row[3]+a:
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
    good = sorted(matches, key = lambda x : x[1].distance)

    # 対応する特徴点同士を描画
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:2], None, flags=2)
    # 結果保存
    cv2.imwrite("./result/"+time_file_name("calc_matches_knn.png"),img3)

    # 特徴量データを取得
    q_kp = []
    t_kp = []
    for p in good[:2]:
        for px in p:
            q_kp.append(kp1[px.queryIdx])
            t_kp.append(kp2[px.trainIdx])

    # 加工対象の画像から特徴点間の角度と距離を計算
    #q_x1, q_y1 = q_kp[0]
    #q_x2, q_y2 = q_kp[-1]
    q_x1, q_y1 = q_kp[0].pt
    q_x2, q_y2 = q_kp[-1].pt

    q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
    q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)

    # テンプレート画像から特徴点間の角度と距離を計算
    #t_x1, t_y1 = t_kp[0]
    #t_x2, t_y2 = t_kp[-1]
    t_x1, t_y1 = t_kp[0].pt
    t_x2, t_y2 = t_kp[-1].pt

    t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
    t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)

    # 切出し位置の計算
    #x1 = q_x1 - t_x1 * (q_len / t_len)
    #x2 = x1 + img2.shape[1] * (q_len / t_len)
    x1 = int(q_x1 - t_x1 * (q_len / t_len))
    x2 = int(x1 + img2.shape[1] * (q_len / t_len))

    #y1 = q_y1 - t_y1 * (q_len / t_len)
    #y2 = y1 + img2.shape[0] * (q_len / t_len)
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
    cv2.imwrite("./result/"+time_file_name("calc_matches_knn_img_rot.png"),img_rot)


# テンプレートマッチを使ったオブジェクト検出
def detect_object_by_matchTemplate(imgcol, img, temp, param):
    arr = []
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
        cv2.rectangle(result2,top_left, bottom_right, (255, 0, 0), 1)
        if len(arr) != 0:
            arr = np.append(arr, [[top_left[0],top_left[1],top_left[0] + w, top_left[1] + h]], axis=0)
        else:
            arr = [[top_left[0],top_left[1],top_left[0] + w, top_left[1] + h]]
    cv2.imwrite("./result/"+time_file_name("detect_object_matchTemplate.png"), result2)

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
    gray = cv2.cvtColor(img00,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),5)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

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
    gray = cv2.cvtColor(img00,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),5)

    # detect
    linesL = lsd(gray)

    return linesL


# 検出した線の描画、保存
def draw_line(img, lines, color_param, file_name):
    result = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 赤線を引く
        result = cv2.line(result, (x1,y1), (x2,y2), color_param, 2)
    cv2.imwrite("./result/"+time_file_name(file_name), result)


# 検出した線の描画、保存（lsd）
def draw_line_lsd(img, lines, color_param, file_name):
    # 検出した線の可視化
    img4 = img.copy()
    for line in linesL:
        x1, y1, x2, y2 = map(int,line[:4])
        if (x2-x1)**2 + (y2-y1)**2 > 1000:
            # 赤線を引く
            img4 = cv2.line(img4, (x1,y1), (x2,y2), color_param, 2)
    cv2.imwrite("./result/"+time_file_name(file_name),img4)

def detect_letter(img, pyocr_param, threshold):
    tools = pyocr.get_available_tools()
    output = []
    if len(tools) == 0:
        return "No OCR tool found"

    tool = tools[pyocr_param["tool"]]
    #res = tool.image_to_string(Image.open(filename), lang=pyocr_param["lang"], builder=pyocr.builders.WordBoxBuilder(tesseract_layout=pyocr_param["layout"]))
    res = tool.image_to_string(img, lang=pyocr_param["lang"], builder=pyocr.builders.WordBoxBuilder(tesseract_layout=pyocr_param["layout"]))
    for letter in res:
        if letter.confidence >= threshold:
            output.append(letter)

    return output


##### main part #####
# 画像データの読み込み, グレースケール化
img1 = cv2.imread("./res/ATRMoji00.png")
img1_gray = cv2.imread("./res/ATRMoji00.png", cv2.IMREAD_GRAYSCALE)
pc = cv2.imread("./res/ATRMoji01.png")
pc_gray = cv2.imread("./res/ATRMoji01.png", cv2.IMREAD_GRAYSCALE)
camera_gray = cv2.imread("./res/camera.png", cv2.IMREAD_GRAYSCALE)
printer_gray = cv2.imread("./res/printer.png", cv2.IMREAD_GRAYSCALE)
experia_gray = cv2.imread("./res/experia.png", cv2.IMREAD_GRAYSCALE)
router_gray = cv2.imread("./res/router.png", cv2.IMREAD_GRAYSCALE)

# Demo: オブジェクト検出＋線検出＋文字検出
# 1. オブジェクト検出
object_arr = []
object_list = [pc_gray, camera_gray, printer_gray, experia_gray, router_gray]
threshold = 0.8
for obj in object_list:
    if len(object_arr) != 0:
        object_arr = np.append(object_arr, detect_object_by_matchTemplate(img1, img1_gray, obj, threshold), axis=0)
    else:
        object_arr = detect_object_by_matchTemplate(img1, img1_gray, obj, threshold)
#print(object_arr)

# 2. 線検出
line_arr = []
linesL = detect_line_by_lsd(img1)
# 後処理+描画：オブジェクト内の場合線と見なさない
img4 = img1.copy()
threshold = 1 # オブジェクト内かどうかの判定用
for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    if (x2-x1)**2 + (y2-y1)**2 > 1000:
        if arrin(x1, y1, x2, y2, object_arr, threshold):
            # 線を引く
            img4 = cv2.line(img4, (x1,y1), (x2,y2), (255,255,255), 5)
            if len(line_arr) != 0:
                line_arr = np.append(line_arr, [[x1, y1, x2, y2]], axis=0)
            else:
                line_arr = [[x1, y1, x2, y2]]
        else:
            pass
            #img4 = cv2.line(img4, (x1,y1), (x2,y2), (0,255,0), 1)
cv2.imwrite("./result/detect_line_lsd2-2.png",img4)
#print(line_arr)

# 3. 文字検出
file_name = "./result/detect_line_lsd2-2.png"
img3 = Image.open(file_name)
# 前処理：画像のリサイズ
resize_param = 3
img3_resized = img3.resize((int(img3.width * resize_param), int(img3.height * resize_param)))
pyocr_param = {"tool":0, "lang":"jpn+eng", "layout":6}
threshold = 80
result_letter = detect_letter(img3_resized, pyocr_param, threshold)
# 後処理：座標の修正(左下原点系と書いてあったが、左上原点系では？もしそうならリサイズ以外の変換不要)
# 描画
if result_letter == 0:
    print("Error: No OCR tool found")
else:
    out = img1.copy()
    img_height,img_width,_ = out.shape
    for d in result_letter:
#        print(d.content, d.position, d.confidence)
        top_left = (int(d.position[0][0] / resize_param), int(d.position[0][1] / resize_param))
        bottom_right = (int(d.position[1][0] / resize_param), int(d.position[1][1] / resize_param))
        cv2.rectangle(out, top_left, bottom_right, (255, 0, 0), 2)
    output_file_name = "detect_letter" + "_resize_" + str(resize_param) + ".png"
    cv2.imwrite("./result/"+time_file_name(output_file_name), out)

# 4. 文字を単語にまとめ上げる（簡易クラスタリング）
# 文字の中心座標
letter_arr_position = []
# 文字
letter_arr_content = []
for d in result_letter:
    if len(letter_arr_position) != 0:
        letter_arr_position = np.append(letter_arr_position, [[(d.position[0][0] + d.position[1][0])/2, (d.position[0][1] + d.position[1][1])/2]], axis=0)
        letter_arr_content = np.append(letter_arr_content, [[d.content]], axis=0)
    else:
        letter_arr_position = [[(d.position[0][0] + d.position[1][0])/2, (d.position[0][1] + d.position[1][1])/2]]
        letter_arr_content = [[d.content]]

result_linkage = linkage(letter_arr_position, method="ward", metric="euclidean")
threshold = 0.15 * np.max(result_linkage[:, 2])
#plt.figure(num=None, figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
#dendrogram(result_linkage, labels=letter_arr_content, color_threshold=threshold)
#plt.show()

result_cluster = fcluster(result_linkage, threshold, criterion='distance')
cluster_num = max(result_cluster)
# 単語
word_arr = []
# 単語の中心座標(文字の座標はresize後の座標であることに注意すること)
word_arr_position = np.zeros((cluster_num, 2))
for j in range(0, cluster_num):
    tmp = ""
    count = 0
    for i in range(0, len(result_cluster)):
        if result_cluster[i] == j+1:
            tmp += letter_arr_content[i][0]
            word_arr_position[j][0] += letter_arr_position[i][0]
            word_arr_position[j][1] += letter_arr_position[i][1]
            count += 1
    word_arr.append(tmp)
    word_arr_position[j][0] = int(word_arr_position[j][0] / (count*resize_param))
    word_arr_position[j][1] = int(word_arr_position[j][1] / (count*resize_param))
#    print(word_arr[j])
#    print(word_arr_position[j])

# 5. 関連づけ
# オブジェクトと文字の位置から関連づけ(中心間の距離が最小となるオブジェクトと関連しているとする)
#print("relation part")
word_text_relations = {}
for i in range(0, len(word_arr_position)):
    distance = 10000
    count = 0
#    print("word:" + word_arr[i])
    for j in range(0, len(object_arr)):
        tmp_x = (object_arr[j][0] + object_arr[j][2]) / 2
        tmp_y = (object_arr[j][1] + object_arr[j][3]) / 2
        d = math.sqrt((tmp_x - word_arr_position[i][0]) ** 2 + (tmp_y - word_arr_position[i][1]) ** 2)
        if distance > d:
            distance = d
            word_text_relations[i] = j
#print(word_text_relations)

# オブジェクトと線の関連づけ T.B.D

# 線とテキストの関連づけ T.B.D

# 6. 出力
# DataLakeに格納する際のMetaData相当の出力
# 仮の構造
# {
#    "objectID":
#        {
#           "position": "x1, y1, x2, y2",
#           "relation_objects": "objectID,objectID,objectID",
#           "text": "hoge"
#        }
# }
relations = {}
for i in range(0, len(object_arr)):
    tmp_position = str(object_arr[i][0]) + "," + str(object_arr[i][1]) + "," + str(object_arr[i][2]) + "," + str(object_arr[i][3])
    keys = [k for k, v in word_text_relations.items() if v == i]
    if len(keys) != 0:
        tmp_text = word_arr[keys[0]]
    else:
        tmp_text = ""
    relations[i] = {"position":tmp_position, "relation_objects":"", "text":tmp_text}
    print(str(i) + " : " + str(relations[i]))

# 画像としての出力
result_demo = img1.copy()
# オブジェクト
for i in range(0, len(object_arr)):
    top_left = (object_arr[i][0], object_arr[i][1])
    bottom_right = (object_arr[i][2], object_arr[i][3])
    keys = [k for k, v in word_text_relations.items() if v == i]
    if len(keys) != 0:
        color_param = (255, 0, 255)
    else:
        color_param = (255, 0, 0)
    cv2.rectangle(result_demo, top_left, bottom_right, color_param, 1)
# 線オブジェクト
for line in line_arr:
    point1 = (line[0], line[1])
    point2 = (line[2], line[3])
    result_demo = cv2.line(result_demo, point1, point2, (0,0,255), 1)
# 文字オブジェクト
for d in result_letter:
    top_left = (int(d.position[0][0] / resize_param), int(d.position[0][1] / resize_param))
    bottom_right = (int(d.position[1][0] / resize_param), int(d.position[1][1] / resize_param))
    cv2.rectangle(result_demo, top_left, bottom_right, (0, 200, 0), 1)

cv2.imwrite("./result/result_demo.png", result_demo)

