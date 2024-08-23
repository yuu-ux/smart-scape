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

    # 画像の平滑化
    blurred1 = cv2.bilateralFilter(gray1, 20, 22, 20)
    blurred2 = cv2.bilateralFilter(gray2, 20, 22, 20)

    # ガウシアンブラーを適応
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # # ガンマ補正
    # gamma = 1.5
    # img1_gamma = adjustGamma(gray1, gamma)
    # img2_gamma = adjustGamma(gray2, gamma)

    return (gray1, gray2)

def main():
    # 画像読み込み
    img1 = cv2.imread('src/image3.png')
    img2 = cv2.imread('src/image4.png')

    # 画像の前処理
    # img1_eq, img2_eq = formatImage(img1, img2)

    # 特徴点を抽出
    kp1, des1 = extractionFeature(img1)
    kp2, des2 = extractionFeature(img2)

    # 画像のマッチング
    matches = matchImage(des1, des2)

    # マッチング画像を表示
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # マッチング点の座標を抽出
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # ホモグラフィを計算
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 画像の変換
    height, width = img2.shape[:2]
    transformed_img = cv2.warpPerspective(img1, H, (width, height))

    # combined_img = cv2.addWeighted(transformed_img, 1.0, img2, 0.3, 0)
    # show(combined_img)
    # 結果をファイルに保存
    output_path = os.path.join('output', 'image.png')
    print(output_path)
    cv2.imwrite(output_path, transformed_img)

    # 保存した画像を読み込む
    output_img = cv2.imread('output/image.png')
    plt.figure(figsize=(18, 8))
    # plt.subplot(1, 2, 1)
    a1 = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    a2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(a1, (21, 21), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(blurred_img)
    blurred_img2 = cv2.GaussianBlur(a2, (21, 21), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img2 = clahe.apply(blurred_img2)
    diff = cv2.absdiff(equalized_img, equalized_img2)
    _, binary_img = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
    H_inv = np.linalg.inv(H)
    transformed_diff = cv2.warpPerspective(binary_img, H_inv, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_NEAREST)

    # 差異データを赤色で表示するための準備
    red_img = np.zeros_like(img1)  # img2と同じサイズのゼロ行列を作成
    red_img[:, :, 2] = transformed_diff    # 赤チャンネルにバイナリ画像を割り当て
    
    # オリジナルの画像と赤い差異データを合成
    combined_img = cv2.addWeighted(img1, 1, red_img, 1, 0)
    cv2.imwrite('output/result.png', combined_img)
    
    # plt.imshow(equalized_img, cmap='gray')
    # plt.title("equalized_img")
    # plt.subplot(1, 2, 2)
    # plt.imshow(equalized_img2, cmap='gray')
    # plt.title("img2")
#     # plt.show()
#     ###########################################
#     # コントラストの強調
#     alpha = 2.0  # コントラストの倍率
#     beta = 1.0     # 明るさの調整（固定）

#     # コントラスト調整
#     adjusted_img = cv2.convertScaleAbs(output_img, alpha=alpha, beta=beta)
#     show(adjusted_img)
#     # img2_cnt = cv2.convertScaleAbs(img2_eq, alpha=alpha, beta=beta)
#     # # 結果を表示
#     # cv2.imshow('Original Image', output_img)
#     # cv2.imshow('Adjusted Image', adjusted_img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     ##############################################
#     diff = cv2.absdiff(adjusted_img, img2_eq)
#     _, binary_img = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY)
#     show(diff)
#     show(binary_img)
#     # edges_img1 = cv2.Canny(adjusted_img, 0, 100)
#     # edges_img2 = cv2.Canny(img2_eq, 0, 100)
#     # intersection = cv2.bitwise_and(edges_img1, edges_img2)
#     # union = cv2.bitwise_or(edges_img1, edges_img2)

#     # # 一致度を計算（Jaccard指数）
#     # similarity_score = np.sum(intersection) / np.sum(union)
#     # print(f"Similarity Score: {similarity_score}")


#     # cv2.imshow("img1", edges_img1)
#     # cv2.imshow("img2", edges_img2)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

#     # # 画像比較
#     # diff = cv2.absdiff(edges_img2, edges_img1)
#     # show(diff)
#     # モルフォロジー操作（クロージング
#     # ノイズの部分が強く出て本質的に出てほしい部分が弱いから厳しいな
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#     # diff_eroded = cv2.erode(diff, kernel, iterations=1)
#     diff_morph = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
#     show(diff_morph)
#     # ノイズフィルタリング（メディアンフィルタ）
#     # diff_filtered = cv2.medianBlur(diff_thresh, 3)

main()
