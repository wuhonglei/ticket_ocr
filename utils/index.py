import cv2
import numpy as np


def get_ordered_rect(contour):
    contour = contour.reshape((4, 2))
    # 一共四个坐标点
    rect = np.zeros((4, 2), dtype=np.float32)

    # 按顺序寻找四个坐标点 [0]:左上 [1]:右上 [2]:右下 [3]:左下
    # 计算左上、右下的和，右上、左下的差
    # 左上角坐标和最小，右下角坐标和最大
    s = np.sum(contour, axis=1)
    rect[0] = contour[np.argmin(s)]
    rect[2] = contour[np.argmax(s)]

    diff = np.diff(contour, axis=1)
    rect[1] = contour[np.argmin(diff)]
    rect[3] = contour[np.argmax(diff)]

    return rect


def get_outline(thresh):
    """ 获取目标轮廓（面积最大的外接轮廓） 
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 闭运算 先膨胀，再腐蚀，可以填充前景物体中的小洞，或者前景物体上的小黑点
    img_closed = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel, iterations=7)

    # 边缘检测
    img_canny = cv2.Canny(img_closed, 100, 200)
    contour, hierarchy = cv2.findContours(
        img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 轮廓排序(面积降序)
    contour = sorted(contour, key=cv2.contourArea, reverse=True)

    # 不规则的矩形轮廓
    outline_contour = contour[0]
    return outline_contour


def get_outline_points(contour):
    """ 将轮廓转为四边形(可能非矩形)  """
    peri = cv2.arcLength(contour, True)
    # 获取轮廓四个角点
    return cv2.approxPolyDP(contour, 0.02 * peri, True)


def get_min_rect(contour):
    """ 获取轮廓的最小外接矩形 """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)
    x, y, w, h = cv2.boundingRect(box)
    rect_contour = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float32)
    size = (w, h)
    return (rect_contour, size)
