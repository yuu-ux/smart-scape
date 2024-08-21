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
height, width = img_3.shape
img_4 = cv2.resize(img_4, (width, height))

# ガンマ補正関数
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma = 1.5
img3_gamma = adjust_gamma(img_3, gamma)
img4_gamma = adjust_gamma(img_4, gamma)

# 特徴点の検出
# AKAZE特徴点検出器の初期化
akaze = cv2.AKAZE_create()

# 特徴点と記述子を検出
kp1, des1 = akaze.detectAndCompute(img3_gamma, None)
kp2, des2 = akaze.detectAndCompute(img4_gamma, None)

# マッチャーの作成 (Brute Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# マッチングを実行
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# マッチング点の座標を抽出
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# ホモグラフィを計算
H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

# 画像の変換
transformed_img = cv2.warpPerspective(img4_gamma, H, (width, height))

# 画像を重ね合わせ
combined_img = cv2.addWeighted(img3_gamma, 0.5, transformed_img, 0.5, 0)

# 結果をファイルに保存
output_path = os.path.join('output', 'image.png')
cv2.imwrite(output_path, combined_img)

# 画像をカラーで読み込む (BGR形式)
output_img = cv2.imread('output/image.png', 0)

diff = cv2.absdiff(img_4, output_img)

# 初期値を設定
threshold = 100

# コールバック関数
def onTrackbar(position):
    global threshold
    threshold = position
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("Simple Threshold", mask)

# ウィンドウを作成
cv2.namedWindow("Simple Threshold")

# トラックバーを作成
cv2.createTrackbar("Track", "Simple Threshold", threshold, 255, onTrackbar)

# 初期の2値化結果を表示
onTrackbar(threshold)

# ループしてトラックバーの調整を待つ
while True:
    # Escキーを押すとループ終了
    if cv2.waitKey(10) == 27:
        break

# ウィンドウを閉じる
cv2.destroyAllWindows()

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
