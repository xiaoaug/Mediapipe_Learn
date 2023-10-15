#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/20 9:57
# @File    : Mediapipe自拍分割模板.py
# @Software: PyCharm
# @Desc    :

import cv2
import mediapipe as mp
import numpy as np
import time


class SelfiSegmentation:
    def __init__(self, model=1):
        """

        :param model: 模型类型0或1。0: 常规  1:横向(更快)
        """

        self.model = model
        self.mp_draw = mp.solutions.drawing_utils  # 引入绘画
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation  # 引入自拍分割模型
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(
            self.model
        )

    def remove_bg(self, img, img_bg=(255, 255, 255), threshold=0.1):
        """去除人像背景

        :param img: 需要去除背景的图片
        :param img_bg: 背景图片，默认黑色背景
        :param threshold: 抠图阈值（0~1），值越大背景消减的越严重
        :return: img_out: 处理好的图像
        """

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB
        results = self.selfie_segmentation.process(img_rgb)  # 获取人像分割结果
        img_bg = cv2.resize(img_bg, (img.shape[1], img.shape[0]))  # 调整背景图片尺寸

        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > threshold
        if isinstance(img_bg, tuple):
            _img_bg = np.zeros(img.shape, dtype=np.uint8)
            _img_bg[:] = img_bg
            img_out = np.where(condition, img, _img_bg)
        else:
            img_out = np.where(condition, img, img_bg)
        return img_out


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 设置视频窗口长度
    cap.set(4, 480)  # 设置视频窗口宽度

    segmentor = SelfiSegmentation()
    img_bg = cv2.imread(
        r"C:\Users\zhang\PycharmProjects\OpenCV\Mediapipe\Resource\background\1.jpg"
    )  # 导入背景图片

    start_time, end_time = time.time(), 0  # 计时，用于计算帧数

    while True:
        success, img = cap.read()

        if not success:
            print("未检测到摄像头")
            break
        if cv2.waitKey(1) in [ord("q"), 27]:
            print("用户主动停止程序")
            break

        img_out = segmentor.remove_bg(img, img_bg=img_bg, threshold=0.1)

        # 计算图像帧数
        end_time = time.time()  # 获取当前时间
        fps = 1 // (end_time - start_time)  # 计算帧数
        start_time = end_time
        cv2.putText(
            img,
            f"FPS: {int(fps)}",
            (0, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

        cv2.imshow("Image", img)
        cv2.imshow("Image Out", img_out)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
