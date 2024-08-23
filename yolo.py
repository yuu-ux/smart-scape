import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# 画像を読み込む
image1_path = 'src/image3.png'
image2_path = 'src/image4.png'

image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# 画像のサイズを揃える
image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# グレースケールに変換
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# ヒストグラム均等化
gray1 = cv2.equalizeHist(gray1)
gray2 = cv2.equalizeHist(gray2)

# ORBを用いた特徴量の抽出とマッチング
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# マッチングした特徴点をリスト化
points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])

# クラスタリングを実行（DBSCANを使用）
db = DBSCAN(eps=50, min_samples=3).fit(points1)

# 各クラスタの領域を特定し、赤丸領域と比較
target_areas = []  # 赤丸領域と重なる部分を保存
for label in np.unique(db.labels_):
    if label == -1:  # ノイズとして無視された点をスキップ
        continue
    mask = (db.labels_ == label)
    cluster_points = points1[mask]
    x, y, w, h = cv2.boundingRect(np.int32(cluster_points))
    
    # 特定の領域（赤丸に対応する領域）のみを検出
    if (100 < x < 500 and 100 < y < 600) or (600 < x < 1100 and 100 < y < 800):
        target_areas.append((x, y, w, h))
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 差分画像を計算してゴミや小さな物体を検出
diff = cv2.absdiff(gray1, gray2)
_, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# 小さな物体（ゴミ）の輪郭を検出
contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 50:  # 小さいノイズを無視するためのフィルタリング
        x, y, w, h = cv2.boundingRect(contour)
        # ゴミが赤丸の領域内にあるか確認
        if (100 < x < 500 and 100 < y < 600) or (600 < x < 1100 and 100 < y < 800):
            cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 結果を表示
cv2.imshow('Detected Changes in Target Areas', image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
