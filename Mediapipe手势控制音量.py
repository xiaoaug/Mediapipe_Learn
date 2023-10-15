#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/15 18:22
# @File    : Mediapipe手势控制音量.py
# @Software: PyCharm
# @Desc    : 基于mediapipe实现手势控制电脑音量的功能

import cv2
import time
import mediapipe
import numpy as np
import math

# 用于调节系统音量所用的库
import pycaw.pycaw  # 调节系统音量的库，pip install pycaw
import ctypes
import comtypes


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
            # for hand_lms in self.results.multi_hand_landmarks:  # 遍历每一只手
            #     if draw:   # 连接手的每个关键点
            #         self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
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
                    (0, 50),
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
        lm_list = []  # 保存手的21个关键点坐标
        if self.results.multi_hand_landmarks:  # 如果检测到手
            hand = self.results.multi_hand_landmarks[hand_no]
            for index, lm in enumerate(hand.landmark):  # 遍历该只手的数据
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z  # 求出该点在图像中的坐标位置
                lm_list.append([cx, cy, cz])
                if draw:
                    cv2.circle(img, (cx, cy), int(w / 50), (255, 0, 255), cv2.FILLED)
        return lm_list


def init_system_volume():
    """初始化系统音量调节任务

    :return: 系统声音输出设备、最小音量值、最大音量值
    """
    # 获取自己的音频设备及其参数
    devices = pycaw.pycaw.AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        pycaw.pycaw.IAudioEndpointVolume._iid_, comtypes.CLSCTX_ALL, None
    )
    volume = ctypes.cast(interface, ctypes.POINTER(pycaw.pycaw.IAudioEndpointVolume))

    # print(volume.GetVolumeRange())  # 音量范围：-65.25, 0, 0.03125
    volume_range = volume.GetVolumeRange()  # 获取音量调节的范围
    min_vol = volume_range[0]  # 最小音量调节值
    max_vol = volume_range[1]  # 最大音量调节值

    return volume, min_vol, max_vol


def change_system_volume(change_volume_value=0, volume=None, min_vol=-65.25, max_vol=0):
    """改变系统音量，音量输入范围：0~100

    :param change_volume_value: 需要改变的音量大小
    :param volume: 当前声音输出设备
    :param min_vol: 系统最小音量大小
    :param max_vol: 系统最大音量大小
    """

    # 由于pycaw音量调节不是线性的，因此直接列出所有的音量指标，直接控制
    volume_dict = {
        0: -65.25,
        1: -56.99,
        2: -51.67,
        3: -47.74,
        4: -44.62,
        5: -42.03,
        6: -39.82,
        7: -37.89,
        8: -36.17,
        9: -34.63,
        10: -33.24,
        11: -31.96,
        12: -30.78,
        13: -29.68,
        14: -28.66,
        15: -27.7,
        16: -26.8,
        17: -25.95,
        18: -25.15,
        19: -24.38,
        20: -23.65,
        21: -22.96,
        22: -22.3,
        23: -21.66,
        24: -21.05,
        25: -20.46,
        26: -19.9,
        27: -19.35,
        28: -18.82,
        29: -18.32,
        30: -17.82,
        31: -17.35,
        32: -16.88,
        33: -16.44,
        34: -16.0,
        35: -15.58,
        36: -15.16,
        37: -14.76,
        38: -14.37,
        39: -13.99,
        40: -13.62,
        41: -13.26,
        42: -12.9,
        43: -12.56,
        44: -12.22,
        45: -11.89,
        46: -11.56,
        47: -11.24,
        48: -10.93,
        49: -10.63,
        50: -10.33,
        51: -10.04,
        52: -9.75,
        53: -9.47,
        54: -9.19,
        55: -8.92,
        56: -8.65,
        57: -8.39,
        58: -8.13,
        59: -7.88,
        60: -7.63,
        61: -7.38,
        62: -7.14,
        63: -6.9,
        64: -6.67,
        65: -6.44,
        66: -6.21,
        67: -5.99,
        68: -5.76,
        69: -5.55,
        70: -5.33,
        71: -5.12,
        72: -4.91,
        73: -4.71,
        74: -4.5,
        75: -4.3,
        76: -4.11,
        77: -3.91,
        78: -3.72,
        79: -3.53,
        80: -3.34,
        81: -3.15,
        82: -2.97,
        83: -2.79,
        84: -2.61,
        85: -2.43,
        86: -2.26,
        87: -2.09,
        88: -1.91,
        89: -1.75,
        90: -1.58,
        91: -1.41,
        92: -1.25,
        93: -1.09,
        94: -0.93,
        95: -0.77,
        96: -0.61,
        97: -0.46,
        98: -0.3,
        99: -0.15,
        100: 0.0,
    }

    change_volume_value = int(change_volume_value)  # 转成整型，方便处理
    # volume_value = np.interp(volume_change_value, [0, 100], [min_vol, max_vol])  # 将输入的音量等比转换为可控的参数
    volume.SetMasterVolumeLevel(volume_dict[change_volume_value], None)  # 调节系统音量
    # print("音量调节到：", change_volume_value)


# 主程序
def main():
    hand_detect = HandDetector(max_hands=1, detection_con=0.7)
    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(3, 640)  # 设置视频窗口长度
    cap.set(4, 480)  # 设置视频窗口宽度
    start_time, end_time = time.time(), 0  # 计时，用于计算帧数
    change_volume_value = 0  # 初始化音量为 0
    volume, min_vol, max_vol = init_system_volume()

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
            x1, y1 = lm_list[4][0], lm_list[4][1]  # 大拇指尖的坐标
            x2, y2 = lm_list[8][0], lm_list[8][1]  # 食指尖的坐标
            line_length = math.hypot(x2 - x1, y2 - y1)  # 计算拇指和食指之间的距离，一般为20~170
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            change_volume_value = np.interp(
                line_length, [30, 180], [0, 100]
            )  # 将20~170的范围等比转换为0~100
            change_system_volume(
                change_volume_value, volume, min_vol, max_vol
            )  # 改变系统音量

        # 显示音量条
        cv2.rectangle(img, (20, 100), (55, 400), (0, 255, 0), 3)  # 音量框
        cv2.rectangle(
            img,
            (55, 400),
            (20, 400 - int(change_volume_value) * 3),
            (0, 255, 0),
            cv2.FILLED,
        )  # 音量条
        cv2.putText(
            img,
            f"{int(change_volume_value)}%",
            (10, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

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

        cv2.imshow("capture", img)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
