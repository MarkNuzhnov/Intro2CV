from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale, hough_circle, hough_circle_peaks
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
import scipy.stats as st

COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}


def count_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    c = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            c += 1
    return c


def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges_city = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 7, 4)

    hough_radii = np.arange(25, 40, 1)
    hough_res = hough_circle(edges_city, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               min_xdistance=150, min_ydistance=150,
                                               total_num_peaks=47)
    centers = list(zip(cy, cx))

    blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
    edges = cv2.adaptiveThreshold(blur, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 27, 13)
    filter_bin = 1 - edges
    # green
    mask_green = (HSV[:, :, 0] > 55) & (HSV[:, :, 0] < 85)
    n_trains = {}
    n_trains['green'] = count_contours(mask_green * filter_bin)

    # blue
    mask_blue = (ycc[:, :, 2] > 150) & (ycc[:, :, 2] < 190)

    n_trains['blue'] = count_contours(mask_blue * filter_bin)

    im_blue = cv2.imread('train/black_red_yellow.jpg')
    blue_template = cv2.cvtColor(im_blue[1058:1088, 1369:1399], cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(img_gray, blue_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.5)
    x_0, y_0 = sorted(list(zip(*loc)))[0]
    blue_no_train = [(x_0, y_0)]
    for (x, y) in sorted(list(zip(*loc))):
        if np.abs(x - x_0) > 50 or np.abs(y - y_0) > 50:
            x_0, y_0 = x, y
            blue_no_train += [(x_0, y_0)]

    n_trains['blue'] = max(0, n_trains['blue'] - len(blue_no_train))

    # yellow
    mask_yellow = (HSV[:, :, 0] > 21) & (HSV[:, :, 0] < 27)

    n_trains['yellow'] = count_contours(mask_yellow * filter_bin)

    im_yellow = cv2.imread('train/black_red_yellow.jpg')
    yellow_template = cv2.cvtColor(im_yellow[903:933, 2139:2170], cv2.COLOR_RGB2GRAY)
    res = cv2.matchTemplate(img_gray, yellow_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.6)
    x_0, y_0 = sorted(list(zip(*loc)))[0]
    yellow_no_train = [(x_0, y_0)]
    for (x, y) in sorted(list(zip(*loc))):
        if np.abs(x - x_0) > 50 or np.abs(y - y_0) > 50:
            x_0, y_0 = x, y
            yellow_no_train += [(x_0, y_0)]

    n_trains['yellow'] = max(0, n_trains['yellow'] - len(yellow_no_train))

    # red
    mask_red = (HSV[:, :, 0] > 177) & (HSV[:, :, 0] < 190)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    opening = cv2.morphologyEx(mask_red * filter_bin, cv2.MORPH_OPEN, kernel)

    n_trains['red'] = count_contours(opening)

    # black
    n_trains['black'] = 0
    mask_black = (HSV[:, :, 2] < 30)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (22, 22))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    mask_black = cv2.morphologyEx(np.uint8(mask_black), cv2.MORPH_CLOSE, kernel1)

    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel2)
    mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel3)

    n_trains['black'] = count_contours(np.uint8(mask_black) * filter_bin)

    scores = {'blue': 0, 'green': 0, 'black': 0, 'yellow': 0, 'red': 0}

    for col in COLORS:
        if n_trains[col] < 3:
            scores[col] = n_trains[col]
        else:
            scores[col] = 1.6 * n_trains[col]  # coefficient was obtained from running different experiments on test images
    print(n_trains)
    return centers, n_trains, scores
