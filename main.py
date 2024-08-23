import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    img1 = cv2.imread('src/image3.png')
    img2 = cv2.imread('src/image4.png')

    # 画像の前処理
    formatedImg1, formatedImg2 = formatImage(img1, img2)

    # 特徴点の抽出からマッチング
    matchedImg, H = matchImage(formatedImg1, formatedImg2)

    # 画像の後処理
    processedImg1, processedImg2 = processImage(matchedImg, formatedImg2)
    
    # 画像の比較・2値化
    diffImg = cv2.absdiff(processedImg1, processedImg2)
    _, binaryImg = cv2.threshold(diffImg, 50, 255, cv2.THRESH_BINARY)

    # オリジナル画像と差異を組み合わせる
    combineOriginal(img1, binaryImg, H)

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

    return (blurred1, blurred2)

def matchImage(img1, img2):
    # 特徴点の検出
    akaze = cv2.AKAZE_create()

    # 特徴点と記述子を検出
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    # マッチャーの作成 (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # マッチングを実行
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 特徴点の座標を格納する配列の初期化
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    # 特徴点の座標を格納
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # ホモグラフィを計算（1のスケールを2に合わせる）
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 画像のスケールを変更する
    height, width = img2.shape[:2]
    transformedImg = cv2.warpPerspective(img1, H, (width, height))
    return (transformedImg, H)

def processImage(img1, img2):
    # ブラー処理
    blurredImg1 = cv2.GaussianBlur(img1, (21, 21), 0)
    blurredImg2 = cv2.GaussianBlur(img2, (21, 21), 0)

    # コントラストの均一化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalizedImg1 = clahe.apply(blurredImg1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalizedImg2 = clahe.apply(blurredImg2)
    return (equalizedImg1, equalizedImg2)
    
def combineOriginal(img1, img2, H):
    # 重ね合わせるための座標を求める
    hInv = np.linalg.inv(H)
    transformedDiff = cv2.warpPerspective(img2, hInv, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_NEAREST)

    # 差異データを赤色で表示するための準備
    redImg = np.zeros_like(img1)
    redImg[:, :, 2] = transformedDiff
    
    # オリジナルの画像と赤い差異データを合成
    combinedImg = cv2.addWeighted(img1, 1, redImg, 1, 0)
    cv2.imwrite('output/result.png', combinedImg)

if __name__ == "__main__": 
    main()