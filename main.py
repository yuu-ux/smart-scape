import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ガンマ補正関数
def adjustGamma(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def show(image, is_gray=False):
    plt.figure(figsize=(8, 5))
    if is_gray:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def extractionFeature(img):
    # 特徴点の検出
    akaze = cv2.AKAZE_create()
    # 特徴点と記述子を検出
    kp, des = akaze.detectAndCompute(img, None)
    return (kp, des)

def matchImage(des1, des2):
    # マッチャーの作成 (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # マッチングを実行
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return (matches)

def formatImage(img1, img2):
    # 画像のリサイズ
    height, width = img1.shape[:2]
    img2 = cv2.resize(img2, (width, height))

    # グレースケール
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def trand_bar(x):
        _, binary_img1 = cv2.threshold(gray1, x, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Image', binary_img1)

    # 画像の平滑化
    blurred1 = cv2.bilateralFilter(gray1, 20, 22, 20)
    blurred2 = cv2.bilateralFilter(gray2, 20, 22, 20)

    # ガンマ補正
    # gamma = 1.5
    # img1_gamma = adjustGamma(gray1, gamma)
    # img2_gamma = adjustGamma(gray2, gamma)

    return (blurred1, blurred2)

def main():
    # 画像読み込み
    img1 = cv2.imread('src/image3.png')
    img2 = cv2.imread('src/image4.png')

    # 画像の前処理
    img1_eq, img2_eq = formatImage(img1, img2)

    # 特徴点を抽出
    kp1, des1 = extractionFeature(img1_eq)
    kp2, des2 = extractionFeature(img2_eq)

    # 画像のマッチング
    matches = matchImage(des1, des2)

    # マッチング画像を表示
    # img_matches = cv2.drawMatches(img1_eq, kp1, img2_eq, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # show(img_matches, gray2, is_gray=True)

    # マッチング点の座標を抽出
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # ホモグラフィを計算
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 画像の変換
    height, width = img2_eq.shape[:2]
    transformed_img = cv2.warpPerspective(img1_eq, H, (width, height))
    show(transformed_img)
    # 画像を重ね合わせ
    # combined_img = cv2.addWeighted(img2_eq, 0.3, transformed_img, 0.7, 0)

    # 結果をファイルに保存
    output_path = os.path.join('output', 'image.png')
    cv2.imwrite(output_path, transformed_img)

    # 保存した画像を読み込む
    output_img = cv2.imread('output/image.png', 0)

    # show(output_img)
    def adjust_threshold(x):
        _, binary_img1 = cv2.threshold(output_img, x, 255, cv2.THRESH_BINARY)
        cv2.imshow('Binary Image', binary_img1)
        # ウィンドウを作成
    cv2.namedWindow('Binary Image')

    # トラックバーを作成
    cv2.createTrackbar('Threshold', 'Binary Image', 150, 255, adjust_threshold)

    # 初期状態の二値化画像を表示
    adjust_threshold(150)

    # キー入力待ち
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # # 2値化
    _, mask = cv2.threshold(diff, 48, 255, cv2.THRESH_BINARY)

main()
