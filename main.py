import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

# 画像読み込み
img1 = cv2.imread('src/image3.png')
img2 = cv2.imread('src/image4.png')

# 画像のリサイズ
height, width = img1.shape[:2]
img2 = cv2.resize(img2, (width, height))

# グレースケール
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ガンマ補正関数
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

gamma = 1.5
img1_gamma = adjust_gamma(gray1, gamma)
img2_gamma = adjust_gamma(gray2, gamma)
# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img1_gamma, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img2_gamma, cmap='gray')
# plt.show()

img1_eq = cv2.equalizeHist(img1_gamma)
img2_eq = cv2.equalizeHist(img2_gamma)
# plt.figure(figsize=(16, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img1_gamma, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img2_gamma, cmap='gray')
# plt.show()

# 特徴点の検出
akaze = cv2.AKAZE_create()

# 特徴点と記述子を検出
kp1, des1 = akaze.detectAndCompute(img1_gamma, None)
kp2, des2 = akaze.detectAndCompute(img2_gamma, None)

# マッチャーの作成 (Brute Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# マッチングを実行
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(img1_gamma, kp1, img2_gamma, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(16, 8))
plt.imshow(img_matches)
plt.show()
# マッチング点の座標を抽出
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt
    points2[i, :] = kp2[match.trainIdx].pt

# ホモグラフィを計算
H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

# 画像の変換
transformed_img = cv2.warpPerspective(img2_gamma, H, (width, height))

# 画像を重ね合わせ
combined_img = cv2.addWeighted(img1_gamma, 0.7, transformed_img, 0.5, 0)

# 結果をファイルに保存
output_path = os.path.join('output', 'image.png')
cv2.imwrite(output_path, combined_img)

# 保存した画像を読み込む
output_img = cv2.imread('output/image.png', 0)

# グレースケールしたgray2と比較する
diff = cv2.absdiff(output_img, gray2)

# plt.figure(figsize=(16, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(gray2, cmap='gray')
# plt.subplot(1, 3, 2)
# plt.imshow(output_img, cmap='gray')
# plt.subplot(1, 3, 3)
# plt.imshow(diff, cmap='gray')
# plt.show()

#  ガウシアンブラーの適応
diff_blurred = cv2.GaussianBlur(diff, (5, 5), 0)

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
