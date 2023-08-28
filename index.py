import cv2
import numpy as np
from pathlib import Path
import pytesseract

from utils import index as utils

img_path = Path(__file__).parent / 'img' / 'page.jpg'
img = cv2.imread(str(img_path))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化
_, img_thresh = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

outline_contour = utils.get_outline(img_thresh)
outline_contour = utils.get_outline_points(outline_contour)

min_rect, size = utils.get_min_rect(outline_contour)

cv2.drawContours(img, [outline_contour], -1, (0, 0, 255), 2)
cv2.drawContours(img, [min_rect.astype(np.int32)], -1, (0, 255, 0), 2)
cv2.imshow('img', img)

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(
    utils.get_ordered_rect(outline_contour), min_rect)

warped_image = cv2.warpPerspective(img_thresh, matrix, size)


text = pytesseract.image_to_string(warped_image, lang='eng')
print(text)

cv2.imshow('warped_image', warped_image)
cv2.waitKey(0)
