import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from collections import Counter

MIN_MATCH_COUNT = 7

FLANN_INDEX_KDTREE = 1
threshold = 0.9


def in_rect(points, h, w):
    x, y = points[1:8:2], points[:7:2]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    h_2, w_2 = x_max - x_min, y_max - y_min
    return np.isclose(h_2 / h, w_2 / w, 0.25)


def search(img, query):
    sift = cv2.SIFT_create()
    key_points_img, descriptors_img = sift.detectAndCompute(img, None)
    key_points_query, description_query = sift.detectAndCompute(query, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=40)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(description_query, descriptors_img, k=2)

    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold * n.distance:
            good.append(m)

    dist = [key_points_img[m.trainIdx].pt for m in good]
    labels = DBSCAN(eps=110, min_samples=3).fit_predict(dist)
    labels_cnt = Counter(labels)
    # print(labels_cnt)
    most_frequent = max(labels_cnt, key=labels_cnt.get)
    cluster = [good[n] for n, label in enumerate(labels) if label == most_frequent]
    if len(cluster) > MIN_MATCH_COUNT:
        src_pts = np.float32([key_points_query[m.queryIdx].pt for m in cluster]).reshape(-1, 1, 2)
        dst_pts = np.float32([key_points_img[m.trainIdx].pt for m in cluster]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = query.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        pts_transformed = np.int32(dst).reshape(8).tolist()
        if not in_rect(pts_transformed, h, w):
            return False, img, None

        img2 = cv2.fillPoly(img, [np.int32(dst)], 0)
        box = pts_transformed[:2] + pts_transformed[4:6]

        return True, img2, box
    else:
        return False, img, None


def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    query = cv2.cvtColor(query, cv2.COLOR_RGB2GRAY)
    new_img = img.copy()

    flag, boxes = True, []
    while flag:
        flag, new_img, box = search(new_img, query)
        if box: boxes.append(box)
    return boxes
