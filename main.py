import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 画像読み込み・グレースケール
# img_1 = cv2.imread('src/image1.png', 0)
# img_2 = cv2.imread('src/image2.png', 0)
img_3 = cv2.imread('src/image3.png', 0)
img_4 = cv2.imread('src/image4.png', 0)

# 画像のリサイズ
height = img_3.shape[0]
width = img_3.shape[1]
img_4 = cv2.resize(img_4, (int(width), int(height)))

# 2値化
def onTrackbar(position):
    global threshold
    threshold = position
# ウィンドウを作成
cv2.namedWindow("Simple Threshold")
cv2.namedWindow("Adaptive Mean Threshold")
cv2.namedWindow("Adaptive Gaussian Threshold")

# トラックバーの初期設定
threshold = 100
cv2.createTrackbar("track", "Simple Threshold", threshold, 255, onTrackbar)

while True:
    # 通常の閾値による二値化
    ret, img_th_simple = cv2.threshold(img_3, threshold, 255, cv2.THRESH_BINARY)
    
    # 適応的二値化（平均値）
    img_th_mean = cv2.adaptiveThreshold(img_3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # 適応的二値化（ガウシアン）
    img_th_gaussian = cv2.adaptiveThreshold(img_3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # ウィンドウに表示
    cv2.imshow("Simple Threshold", img_th_simple)
    cv2.imshow("Adaptive Mean Threshold", img_th_mean)
    cv2.imshow("Adaptive Gaussian Threshold", img_th_gaussian)

    # Escキーを押すとループ終了
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()

# diff = cv2.absdiff(img_3, img_4)
# diff = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)
# plt.imshow(result)
# plt.show()

# # ORB検出器を初期化
# orb = cv2.ORB_create()

# # 各画像の特徴点と記述子を検出
# kp1, des1 = orb.detectAndCompute(img_1_gray, None)
# kp2, des2 = orb.detectAndCompute(img_2_gray, None)

# # 特徴点を画像に描画（画像1についてのみ）
# img1_kp = cv2.drawKeypoints(img_1, kp1, None, color=(0, 255, 0), flags=0)

# # 画像を引き算
# img_diff = cv2.absdiff(img_1_gray, img_2_gray)

# # 2値化
# ret2, img_th = cv2.threshold(img_diff, 20, 255, cv2.THRESH_BINARY)

# # 輪郭を検出
# contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 閾値以上の差分を四角で囲う
# for i, cnt in enumerate(contours):
#     x, y, width, height = cv2.boundingRect(cnt)
#     if width > 20 or height > 20:
#         cv2.rectangle(img1_kp, (x, y), (x + width, y + height), (0, 0, 255), 1)

# # 画像表示の設定
# plt.figure(figsize=(10, 3))  # ウィンドウサイズを設定
# plt.subplots_adjust(wspace=0.05, hspace=0)  # サブプロット間の間隔を設定

# # 元画像1を表示
# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))

# # 元画像2を表示
# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))

# # 比較画像を表示
# plt.subplot(1, 3, 3)
# plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))

# plt.show()  # 画像を表示
