import cv2
import numpy as np
from pathlib import Path
import pytesseract

img_path = Path(__file__).parent / 'img' / '中文小票.jpg'
img = cv2.imread(str(img_path))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 二值化
_, img_thresh = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


def get_outline(thresh):
    # 闭运算 先膨胀，再腐蚀，可以填充前景物体中的小洞，或者前景物体上的小黑点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_closed = cv2.morphologyEx(
        img_thresh, cv2.MORPH_CLOSE, kernel, iterations=7)

    # 边缘检测
    img_canny = cv2.Canny(img_closed, 100, 200)
    contour, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img.shape[:2]
    area = height * width

    # 轮廓排序(面积降序)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)

    # 不规则的矩形轮廓
    outline_contour = contour[0]
    if cv2.contourArea(outline_contour) < area * 0.5:
        return None

    return outline_contour


outline_contour = get_outline(img_thresh)
if outline_contour is None:
    custom_tessdata_dir = '/opt/homebrew/lib/python3.11/site-packages/pytesseract/tessdata'
    text = pytesseract.image_to_string(
        img_thresh, lang='chi_sim+eng', config=f'--tessdata-dir {custom_tessdata_dir}')
    print(text)
    exit()

# 获取轮廓四个角点
outline_contour = cv2.approxPolyDP(outline_contour, 40, True)

# 获取第一个轮廓的最小包围矩形
rect = cv2.minAreaRect(outline_contour)
center, size, angle = rect

# 绘制图像和包围矩形
box = cv2.boxPoints(rect)
box = box.astype(np.int32)

print(outline_contour)
print('---')
print(box)


# 透视变换
# 透视变换的原理是通过变换矩阵将原图中的四边形区域映射为矩形
# 透视变换需要四个点，这四个点需要按照左上、右上、右下、左下的顺序排列
original_points = np.array([
    outline_contour[1][0],
    outline_contour[0][0],
    outline_contour[3][0],
    outline_contour[2][0],
])


target_points = box.astype(np.float32)

# 计算透视变换矩阵
matrix = cv2.getPerspectiveTransform(
    original_points.astype(np.float32), target_points)

dsize = img.shape[:2][::-1]  # (width, height)
# 执行透视变形
warped_image = cv2.warpPerspective(img_thresh, matrix, dsize)

rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, scale=1)
# 进行图像旋转, 设置填充颜色为白色
rotated_image = cv2.warpAffine(
    warped_image, rotation_matrix, dsize, borderValue=(255, 255, 255))

rotated_corners = cv2.transform(np.array([target_points]), rotation_matrix)[
    0]  # 使用旋转变换矩阵来旋转角点

x, y, w, h = cv2.boundingRect(rotated_corners)
ticket = rotated_image[y:y + h, x:x + w]
text = pytesseract.image_to_string(ticket)
print(text)
cv2.imshow('ticket', ticket)
cv2.waitKey(0)
