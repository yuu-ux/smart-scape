import cv2
import numpy as np

def detect_differences(img1_path, img2_path):
    # 画像を読み込む
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 画像のサイズを取得
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 画像のサイズを比較してリサイズする
    if (height1, width1) != (height2, width2):
        # img1をimg2のサイズにリサイズ
        img1 = cv2.resize(img1, (width2, height2))

    # 画像をグレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーを適用
    gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # 差分を計算
    diff = cv2.absdiff(gray1, gray2)

    # 差分に閾値を適用（閾値を調整）
    thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]

    # モルフォロジー演算を強化
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 輪郭を検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 結果を表示する画像
    result = img1.copy()

    # 各輪郭に対して
    for contour in contours:
        # 面積が小さすぎる輪郭は無視（最小サイズを増加）
        if cv2.contourArea(contour) < 1000:  # この値を調整
            continue
        
        # 矩形を描画
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return result

# 画像のパス
img1_path = 'src/image3.png'
img2_path = 'src/image4.png'

# 差分を検出
result = detect_differences(img1_path, img2_path)

# 結果を保存
cv2.imwrite('result_improved.jpg', result)
