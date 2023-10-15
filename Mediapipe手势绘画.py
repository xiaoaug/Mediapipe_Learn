#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/16 10:25
# @File    : Mediapipe手势绘画.py
# @Software: PyCharm
# @Desc    : 基于Mediapipe的手势绘画功能

import cv2
import time
import mediapipe
import numpy as np


# 手检测
class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """
        初始化参数
        mode: 是否输入静态图像
        :param max_hands: 检测到手的最大数量
        :param detection_con: 检测手的置信度
        :param track_con: 追踪手的置信度
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.results = None  # 保存检测手部位置的结果

        self.mp_hands = mediapipe.solutions.hands  # 引入手部模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )
        self.mp_draw = mediapipe.solutions.drawing_utils  # 引入绘画，用于画出手部位置
        self.connect_style = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2
        )  # 设定连线的颜色参数
        self.landmark_style = self.mp_draw.DrawingSpec(
            color=(0, 0, 255), thickness=4
        )  # 设定画出手位置的颜色参数

        self.tip_ids = [4, 8, 12, 16, 20]  # 手指的指尖id
        self.lm_list = []  # 保存手的21个关键点坐标

    def find_hands(self, img, draw=True):
        """
        检测手掌
        :param img:要识别的一帧图像
        :param draw:是否对手的标志点进行绘图
        :return:绘画完成的一帧图像
        """
        img = cv2.flip(img, 1)  # 图像水平反转（镜像）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图像BGR转RGB
        self.results = self.hands.process(img_rgb)  # 检测手的位置
        hand_lms = self.results.multi_hand_landmarks  # 手的标记点
        handedness = self.results.multi_handedness  # 区分左右手
        if hand_lms:  # 如果检测到手
            handedness_info = ""  # 用于在图像中显示左手和右手数据
            for index in range(len(hand_lms)):  # 检索每一只识别出来的手
                # 记录左右手的信息
                handedness_label = (
                    handedness[index].classification[0].label
                )  # 读取该只手是左手还是右手
                handedness_score = handedness[index].classification[0].score  # 手的置信度
                handedness_info += (
                    f"{index}-{handedness_label}-{handedness_score:.2f}  "  # 用于显示
                )
                cv2.putText(
                    img,
                    handedness_info,
                    (0, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                if draw:  # 连接手的每个关键点
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_lms[index],
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_style,
                        connection_drawing_spec=self.connect_style,
                    )
        return img  # 返回处理完成的图像

    def find_position(self, img, hand_no=0, draw=True):
        """
        检测手的标志点
        :param img: 要识别的一帧图像
        :param hand_no: 手的编号
        :param draw: 是否对手的标志点进行个性化绘图
        :return: 手的21个标志点位置
        """
        self.lm_list = []
        if self.results.multi_hand_landmarks:  # 如果检测到手
            hand = self.results.multi_hand_landmarks[hand_no]
            for index, lm in enumerate(hand.landmark):  # 遍历该只手的数据
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z  # 求出该点在图像中的坐标位置
                self.lm_list.append([index, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def fingers_up(self):
        """
        检测手指向上的数量
        :return: 五个手指是否向上竖着【0：没有  1：有】
        """
        fingers = []
        # 拇指向上判断
        if (
            self.lm_list[self.tip_ids[0] - 1][1] - 15
            < self.lm_list[self.tip_ids[0]][1]
            < self.lm_list[self.tip_ids[0] - 1][1] + 15
            and self.lm_list[self.tip_ids[0]][2] < self.lm_list[self.tip_ids[0] - 1][2]
        ):
            fingers.append(1)
        else:
            fingers.append(0)

        # 其他四个手指向上判断
        for index in range(1, 5):
            if (
                self.lm_list[self.tip_ids[index] - 1][1] - 15
                < self.lm_list[self.tip_ids[index]][1]
                < self.lm_list[self.tip_ids[index] - 1][1] + 15
                and self.lm_list[self.tip_ids[index]][2]
                < self.lm_list[self.tip_ids[index] - 2][2]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


# 主函数
def main():
    hand_detect = HandDetector(max_hands=1, detection_con=0.7)
    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(3, 640)  # 设置视频窗口长度
    cap.set(4, 480)  # 设置视频窗口宽度
    start_time, end_time = time.time(), 0  # 计时，用于计算帧数
    x1_pre, y1_pre = 0, 0  # 画图所用的点，记录上一次x1, y1的点
    img_canvas = np.zeros((480, 640, 3), np.uint8)  # 创建一个画布，用于画画和显示所画的图像

    while True:
        success, img = cap.read()

        if not success:
            print("未检测到摄像头")
            break
        if cv2.waitKey(1) in [ord("q"), 27]:
            print("用户主动停止程序")
            break

        img = hand_detect.find_hands(img=img)  # 寻找手是否存在
        lm_list = hand_detect.find_position(img=img, draw=False)  # 找到手的位置信息
        if lm_list:  # 获取到手的位置信息
            x1, y1 = lm_list[8][1:]  # 食指尖端坐标
            x2, y2 = lm_list[12][1:]  # 中指尖端坐标

            fingers = hand_detect.fingers_up()  # 检测有几根手指向上
            # 只有食指竖起来的情况
            if fingers[1] == 1 and fingers[2] == fingers[3] == fingers[4] == 0:
                print("按下画笔")
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.putText(
                    img,
                    "Pen Down",
                    (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                if x1_pre == y1_pre == 0:
                    x1_pre, y1_pre = x1, y1
                cv2.line(img, (x1_pre, y1_pre), (x1, y1), (255, 0, 255), 10)
                cv2.line(img_canvas, (x1_pre, y1_pre), (x1, y1), (255, 0, 255), 10)
                x1_pre, y1_pre = x1, y1
            # 只有食指和中指竖起来的情况
            if fingers[1] == fingers[2] == 1 and fingers[3] == fingers[4] == 0:
                print("拿起画笔")
                x1_pre, y1_pre = 0, 0  # 还原先前的点
                cv2.circle(img, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
                cv2.putText(
                    img,
                    "Pen Up",
                    (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
            # 只有食指中指无名指竖起来的情况
            if fingers[1] == fingers[2] == fingers[3] == 1 and fingers[4] == 0:
                print("使用擦除工具")
                cv2.circle(img, (x2, y2), 20, (255, 255, 255), cv2.FILLED)
                cv2.circle(img_canvas, (x2, y2), 20, (0, 0, 0), cv2.FILLED)
                cv2.putText(
                    img,
                    "Eraser",
                    (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
        else:
            x1_pre, y1_pre = 0, 0  # 还原先前的点

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
            (255, 255, 255),
            2,
        )

        # 处理画布，并融合进真实的图像中
        img_canvas_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(
            img_canvas_gray, 50, 255, cv2.THRESH_BINARY_INV
        )  # 图像转成二进制
        dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, dst)
        img = cv2.bitwise_or(img, img_canvas)

        cv2.imshow("capture", img)
        # cv2.imshow('test', img_canvas)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
