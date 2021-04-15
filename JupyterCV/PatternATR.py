#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import math
import pickle
import numpy as np
import time


# In[9]:


get_ipython().system("pip install 'ocrd-fork-pylsd == 0.0.3'")
import time
from pylsd.lsd import lsd


# In[8]:


from IPython.display import display, Image

def display_cv_image(image, format='.png'):
    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
    display(Image(data=decoded_bytes))
    
def time_file_name(fname):
    return time.strftime("%d_%b_%Y_", time.gmtime()) + fname


# In[6]:


time.strftime("%d_%b_%Y", time.gmtime())


# In[10]:


# 画像読込
img1 = cv2.imread("./res/ATRMoji00.png")
img2 = cv2.imread("./res/ATRMoji01.png")

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
ratio = 0.2
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 特徴量をマッチング状況に応じてソート
good = sorted(matches, key = lambda x : x[1].distance)

# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:2], None, flags=2)

display_cv_image(img3, '.png')


# In[11]:


# 画像読込
img1 = cv2.imread("./res/ATRMoji00.png")
img2 = cv2.imread("./res/ATRMoji01.png")

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
ratio = 0.2
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 特徴量をマッチング状況に応じてソート
good = sorted(matches, key = lambda x : x[1].distance)

# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:2], None, flags=2)

display_cv_image(img3, '.png')

# 特徴量データを取得

q_kp = []
t_kp = []



for p in good[:2]:
    for px in p:
        q_kp.append(kp1[px.queryIdx])
        t_kp.append(kp2[px.trainIdx])

# 加工対象の画像から特徴点間の角度と距離を計算
q_x1, q_y1 = q_kp[0]
q_x2, q_y2 = q_kp[-1]

q_deg = math.atan2(q_y2 - q_y1, q_x2 - q_x1) * 180 / math.pi
q_len = math.sqrt((q_x2 - q_x1) ** 2 + (q_y2 - q_y1) ** 2)

# テンプレート画像から特徴点間の角度と距離を計算
t_x1, t_y1 = t_kp[0]
t_x2, t_y2 = t_kp[-1]

t_deg = math.atan2(t_y2 - t_y1, t_x2 - t_x1) * 180 / math.pi
t_len = math.sqrt((t_x2 - t_x1) ** 2 + (t_y2 - t_y1) ** 2)

# 切出し位置の計算
x1 = q_x1 - t_x1 * (q_len / t_len)
x2 = x1 + img2.shape[1] * (q_len / t_len)

y1 = q_y1 - t_y1 * (q_len / t_len)
y2 = y1 + img2.shape[0] * (q_len / t_len)

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
display_cv_image(img_rot, '.png')


# kokokara

# In[13]:


#coding:utf-8
import cv2
#画像をグレースケールで読み込む
imgcol = cv2.imread("./res/ATRMoji00.png",1)
img = cv2.imread("./res/ATRMoji00.png",0)
temp = cv2.imread("./res/ATRMoji01.png", 0)
#マッチングテンプレートを実行
#比較方法はcv2.TM_CCOEFF_NORMEDを選択
result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
#検出結果から検出領域の位置を取得
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
w, h = temp.shape[::-1]
bottom_right = (top_left[0] + w, top_left[1] + h)
#検出領域を四角で囲んで保存
result = cv2.imread("./res/ATRMoji00.png")
cv2.rectangle(result,top_left, bottom_right, (255, 0, 0), 2)
display_cv_image(result, '.png')
cv2.imwrite("./res/result.png", result)


# In[16]:


imgH = cv2.imread("./res/ATRMoji00.png")


# In[17]:


gray = cv2.cvtColor(imgH, cv2.COLOR_BGR2GRAY)
display_cv_image(gray, '.png')
#cv2.imwrite("./res/ATRMoji00G.png", gray)


# In[18]:


gray2 = cv2.bitwise_not(gray)
display_cv_image(gray2, '.png')
#cv2.imwrite("./res/ATRMoji00HAN.png", gray)


# kokomade

# In[12]:


lines = cv2.HoughLinesP(gray2, rho=1, theta=np.pi/360, threshold=80, minLineLength=250, maxLineGap=5)
print(lines)
for line in lines:
    x1, y1, x2, y2 = line[0]
    # 赤線を引く
    red_line_img = cv2.line(imgcol, (x1,y1), (x2,y2), (0,200,255), 3)
    cv2.imwrite("./res/ATRMoji00HAFU2999.png", red_line_img)
display_cv_image(red_line_img, '.png')


# In[28]:


img00 = imgcol.copy()
#img00 = cv2.resize(img00,(int(img00.shape[1]/5),int(img00.shape[0]/5)))
gray = cv2.cvtColor(img00,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),5)

t1 = time.time()
edges = cv2.Canny(gray,50,150,apertureSize = 3)
linesH = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=50, minLineLength=50, maxLineGap=10)
t2 = time.time()

linesL = lsd(gray)
t3 = time.time()

img2 = img00.copy()
for line in linesH:
    x1, y1, x2, y2 = line[0]

    # 赤線を引く
    img2 = cv2.line(img2, (x1,y1), (x2,y2), (0,0,255), 3)

cv2.imwrite('samp_hagh.jpg',img2)
img3 = img00.copy()
img4 = img00.copy()
for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    img3 = cv2.line(img3, (x1,y1), (x2,y2), (0,0,255), 3)
    if (x2-x1)**2 + (y2-y1)**2 > 1000:
       # 赤線を引く
       img4 = cv2.line(img4, (x1,y1), (x2,y2), (0,0,255), 3)
print("Hagh")
print(len(linesH),"lines")
print(t2-t1,"sec")
print("time per a line :{:.4f}".format((t2-t1)/len(linesH)))
print("LSD")
print(len(linesL),"lines")
print(t3-t2,"sec")
print("time per a line {:.4f}".format((t3-t2)/len(linesL)))
cv2.imwrite('samp_pylsd.jpg',img3)
cv2.imwrite('samp_pylsd2.jpg',img4)


# In[34]:


#画像をグレースケールで読み込む
imgcol = cv2.imread("./res/ATRMoji00.png",1)
img = cv2.imread("./res/ATRMoji00.png",0)
temp = cv2.imread("./res/ATRMoji01.png", 0)

#マッチングテンプレートを実行
result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
print(result)
#類似度の設定(0~1)
threshold = 0.8
#検出結果から検出領域の位置を取得
loc = np.where(result >= threshold)
#検出領域を四角で囲んで保存
result = cv2.imread("./res/ATRMoji00.png")
w, h = temp.shape[::-1]
for top_left in zip(*loc[::-1]):
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(result,top_left, bottom_right, (255, 0, 0), 2)
    np.append(a_2d, a_2d_ex, axis=1)
display_cv_image(result, '.png')
cv2.imwrite("result2.png", result)


# In[26]:


def arrin(x1,y1,x2,y2,array):
    for row in array:
        if x1>row[0] and y1>row[1]:
            print(x1)
            print(y1)
            return 0
        if x2<row[2] and y2<row[3]:
            print(x2)
            print(y2)
            return 0
    print (array)
    return 1
            


# In[32]:


def arrin(x1,y1,x2,y2,array):
    for row in array:
        if x1>row[0] and y1>row[1] and x1<row[2] and y1<row[3]:
            return 0
        if x2>row[0] and y2>row[1] and x2<row[2] and y2<row[3]:
            return 0
    #print (array)
    return 1


# In[29]:


def arrin(x1,y1,x2,y2,array):
    a = 10
    for row in array:
        if x1>=row[0]-a and y1>=row[1]-a and x1<=row[2]+a and y1<=row[3]+a:
            if x2>=row[0]-a and y2>=row[1]-a and x2<=row[2]+a and y2<=row[3]+a:
                return 0
    #print (array)
    return 1


# In[23]:


arr = np.array([[2,2,4,4]])
#画像をグレースケールで読み込む
imgcol = cv2.imread("./res/ATRMoji00.png",1)
img = cv2.imread("./res/ATRMoji00.png",0)
temp = cv2.imread("./res/ATRMoji01.png", 0)

#マッチングテンプレートを実行
result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
print(result)
#類似度の設定(0~1)
threshold = 0.8
#検出結果から検出領域の位置を取得
loc = np.where(result >= threshold)
#検出領域を四角で囲んで保存
result = cv2.imread("./res/ATRMoji00.png")
w, h = temp.shape[::-1]
for top_left in zip(*loc[::-1]):
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(result,top_left, bottom_right, (255, 0, 0), 2)
    print([top_left[0],top_left[1],top_left[0] + w, top_left[1] + h])
    arr = np.append(arr, [[top_left[0],top_left[1],top_left[0] + w, top_left[1] + h]], axis=0)
display_cv_image(result, '.png')
print (arr)


# In[30]:



img00 = imgcol.copy()
#img00 = cv2.resize(img00,(int(img00.shape[1]/5),int(img00.shape[0]/5)))
gray = cv2.cvtColor(img00,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),5)

t1 = time.time()
edges = cv2.Canny(gray,50,150,apertureSize = 3)
linesH = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=50, minLineLength=50, maxLineGap=10)
t2 = time.time()

linesL = lsd(gray)
t3 = time.time()

img2 = img00.copy()
for line in linesL:
    x1, y1, x2, y2 =  map(int,line[:4])
    if arrin(x1, y1, x2, y2,arr):
        # 赤線を引く
        #print(str(x1)+":"+str(y1)+":"+str(x2)+":"+str(y2))
        img2 = cv2.line(img2, (x1,y1), (x2,y2), (0,0,255), 3)
    else :
        #print(str(x1)+":"+str(y1)+":"+str(x2)+":"+str(y2))
        img2 = cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), 3)

cv2.imwrite('samp_haghJOUKEN-9.jpg',img2)
img3 = img00.copy()
img4 = img00.copy()
for line in linesL:
    x1, y1, x2, y2 = map(int,line[:4])
    img3 = cv2.line(img3, (x1,y1), (x2,y2), (0,0,255), 2)
    if (x2-x1)**2 + (y2-y1)**2 > 1000:
        if arrin(x1, y1, x2, y2,arr):
            # 赤線を引く
            #print(str(x1)+":"+str(y1)+":"+str(x2)+":"+str(y2))
            img4 = cv2.line(img2, (x1,y1), (x2,y2), (0,0,255), 3)
        else :
            #print(str(x1)+":"+str(y1)+":"+str(x2)+":"+str(y2))
            img4 = cv2.line(img2, (x1,y1), (x2,y2), (0,255,0), 3)
print("Hagh")
print(len(linesH),"lines")
print(t2-t1,"sec")
print("time per a line :{:.4f}".format((t2-t1)/len(linesH)))
print("LSD")
print(len(linesL),"lines")
print(t3-t2,"sec")
print("time per a line {:.4f}".format((t3-t2)/len(linesL)))

cv2.imwrite(time_file_name("lsd2-2.jpg"),img2)


# In[ ]:





# In[ ]:




