from PIL import Image
import numpy as np
import cv2 as cv


def seg_lane_by_cv(img):
    # 获得行数和列数即图片大小
    rowNum, colNum = img.shape[:2]

    for x in range(rowNum):
        for y in range(colNum):
            cur = img[x, y].tolist()
            if (cur[0] < 115 or cur[0] > 150) or (cur[1] < 130 or cur[1] > 160) or \
                    (cur[2] < 140 or cur[2] > 170) or (rowNum / 1.8 > x) or \
                    (y < colNum / 3) or (y > colNum / 1.5):
                img[x, y] = np.array([0, 0, 0])

    # 保存修改后图片
    cv.imwrite('b1.png', img)


if __name__ == '__main__':
    im = cv.imread('1_rgb.png')
    seg_lane_by_cv(im)
