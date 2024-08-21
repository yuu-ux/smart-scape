import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

# 画像読み込み・グレースケール
# img_1 = cv2.imread('src/image1.png', 0)
# img_2 = cv2.imread('src/image2.png', 0)
img_3 = cv2.imread('src/image3.png', 0)
img_4 = cv2.imread('src/image4.png', 0)

# 画像のリサイズ
height = img_3.shape[0]
width = img_3.shape[1]
img_4 = cv2.resize(img_4, (int(width), int(height)))

# ガンマ補正
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# 元画像のヒストグラムを計算
hist_img3_before = cv2.calcHist([img_3], [0], None, [256], [0, 256])
hist_img4_before = cv2.calcHist([img_4], [0], None, [256], [0, 256])

# ガンマ補正
gamma = 1.5
img3_gamma = adjust_gamma(img_3, gamma)
img4_gamma = adjust_gamma(img_4, gamma)

# ガンマ補正後のヒストグラムを計算
hist_img3_after = cv2.calcHist([img3_gamma], [0], None, [256], [0, 256])
hist_img4_after = cv2.calcHist([img4_gamma], [0], None, [256], [0, 256])

# ヒストグラムと画像を並べてプロット
plt.figure(figsize=(16, 8))

# ガンマ補正前のヒストグラム
plt.subplot(3, 2, 1)
plt.plot(hist_img3_before, color='blue', label='img3 before gamma')
plt.plot(hist_img4_before, color='red', label='img4 before gamma')
plt.title('Histogram Before Gamma Correction')
plt.xlabel('Brightness Value')
plt.ylabel('Frequency')
plt.legend()

# ガンマ補正後のヒストグラム
plt.subplot(3, 2, 2)
plt.plot(hist_img3_after, color='blue', label='img3 after gamma (1.5)')
plt.plot(hist_img4_after, color='red', label='img4 after gamma (1.5)')
plt.title('Histogram After Gamma Correction')
plt.xlabel('Brightness Value')
plt.ylabel('Frequency')
plt.legend()

# 元の画像とガンマ補正後の画像を表示
plt.subplot(3, 2, 3)
plt.imshow(cv2.cvtColor(img_3, cv2.COLOR_GRAY2RGB))
plt.title('Original Image 3')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.imshow(cv2.cvtColor(img3_gamma, cv2.COLOR_GRAY2RGB))
plt.title('Gamma Corrected Image 3 (1.5)')
plt.axis('off')

# 元の画像とガンマ補正後の画像を表示
plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(img_4, cv2.COLOR_GRAY2RGB))
plt.title('Original Image 4')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.imshow(cv2.cvtColor(img4_gamma, cv2.COLOR_GRAY2RGB))
plt.title('Gamma Corrected Image 4 (1.5)')
plt.axis('off')

plt.tight_layout()
plt.show()


# # コントラスト向上
# # 特徴点の検出
# # harris
# # img_3がグレースケールの場合、まずカラーに変換
# if len(img_3.shape) == 2:  # グレースケール画像か確認
#     img_harris = cv2.cvtColor(copy.deepcopy(img_3), cv2.COLOR_GRAY2BGR)
# else:
#     img_harris = copy.deepcopy(img_3)

# # Harrisコーナー検出
# img_dst = cv2.cornerHarris(img_3, 10, 3, 0.04)

# # 結果をもとに赤色でコーナーを強調
# img_harris[img_dst > 0.05 * img_dst.max()] = [0, 0, 255]

# # 画像を表示 (RGB形式に変換して表示)
# plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
# plt.axis('off')  # 軸を非表示にする（任意）
# plt.show()

# # 画像を引き算
# diff = cv2.absdiff(img_3, img_4)

# # 2値化
# # 適応的二値化（平均値）
# # これがよさそうだけど、煮詰まったら確認の余地あり
# res = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# masked_diff = cv2.bitwise_and(diff, res)
# # diff = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
# plt.imshow(masked_diff, cmap='gray')
# plt.axis('off')
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
