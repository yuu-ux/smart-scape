import cv2
import os
from opencv_japanese import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

# 画像読み込み
img_1 = imread(os.path.join(dirname, 'image3.png'))
img_2 = imread(os.path.join(dirname, 'image4.png'))

height = img_2.shape[0]
width = img_2.shape[1]

img_1 = cv2.resize(img_1, (int(width), int(height)))

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

# ORB検出器を初期化
orb = cv2.ORB_create()

# 各画像の特徴点と記述子を検出
kp1, des1 = orb.detectAndCompute(img_1_gray, None)
kp2, des2 = orb.detectAndCompute(img_2_gray, None)

# 特徴点を画像に描画（画像1についてのみ）
img1_kp = cv2.drawKeypoints(img_1, kp1, None, color=(0, 255, 0), flags=0)

# 画像を引き算
img_diff = cv2.absdiff(img_1_gray, img_2_gray)

# 2値化
ret2, img_th = cv2.threshold(img_diff, 20, 255, cv2.THRESH_BINARY)

# 輪郭を検出
contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 閾値以上の差分を四角で囲う
for i, cnt in enumerate(contours):
    x, y, width, height = cv2.boundingRect(cnt)
    if width > 20 or height > 20:
        cv2.rectangle(img1_kp, (x, y), (x + width, y + height), (0, 0, 255), 1)

# 画像表示の設定
plt.figure(figsize=(10, 3))  # ウィンドウサイズを設定
plt.subplots_adjust(wspace=0.05, hspace=0)  # サブプロット間の間隔を設定

# 元画像1を表示
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))

# 元画像2を表示
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))

# 比較画像を表示
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))

plt.show()  # 画像を表示
