#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/18 16:01
# @File    : Mediapipe虚拟图像拖放.py
# @Software: PyCharm
# @Desc    :

import cv2
import time
import mediapipe
import math


# 手检测
class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        """初始化参数

        :param mode: 是否输入静态图像
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

    def find_hands(self, img, draw=True, flip=True):
        """检测手掌

        :param img: 要识别的一帧图像
        :param draw: 是否对手的标志点进行绘图
        :param flip: 是否翻转图像
        :return: all_hands: 所有手的信息（type、lms_list、box_size、center）
                 img: 返回处理好的图像
        """

        # 图像处理
        if flip:
            img = cv2.flip(img, 1)  # 图像水平反转（镜像）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图像BGR转RGB
            self.results = self.hands.process(img_rgb)  # 检测手的位置
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 图像BGR转RGB
            self.results = self.hands.process(img_rgb)  # 检测手的位置

        all_hands = []  # 记录左右手是否出现
        h, w, c = img.shape  # 该帧图像的高、宽、标志（用不到）

        # 如果检测到手
        if self.results.multi_hand_landmarks:
            # 遍历手的类型、手的关键点
            for hand_type, hand_lms in zip(
                self.results.multi_handedness, self.results.multi_hand_landmarks
            ):
                hand = {}  # 保存手的各种信息
                lms_list = []  # 保存21个关键点的坐标
                lms_x_list = []  # 保存21个关键点的x坐标
                lms_y_list = []  # 保存21个关键点的y坐标

                # 遍历该只手的21个关键点
                for index, lms in enumerate(hand_lms.landmark):
                    x, y, z = int(lms.x * w), int(lms.y * h), lms.z  # 该点的坐标
                    lms_list.append([x, y, z])  # 保存该坐标值
                    lms_x_list.append(x)  # 保存该关键点的x坐标
                    lms_y_list.append(y)  # 保存该关键点的y坐标

                # 检测是左手还是右手
                if flip:  # 图像如果翻转，则能正确识别左右手信息
                    hand["type"] = hand_type.classification[0].label
                else:  # 图像没翻转，左右手信息获取是相反的，因此需要数据交换一下
                    if hand_type.classification[0].label == "Right":
                        hand["type"] = "Left"
                    else:
                        hand["type"] = "Right"

                # 计算方框的参数
                x_min, x_max = min(lms_x_list), max(lms_x_list)  # 21个关键点的最小、最大x坐标值
                y_min, y_max = min(lms_y_list), max(lms_y_list)  # 21个关键点的最小、最大y坐标值
                box_w, box_h = (x_max - x_min), (y_max - y_min)  # 方框的宽度和高度
                box_size = (x_min, y_min, box_w, box_h)  # 储存该方框的大小参数，用于画图
                x_center, y_center = box_size[0] + (box_size[2] // 2), box_size[1] + (
                    box_size[3] // 2
                )  # 手的中心点

                hand["lms_list"] = lms_list  # 21个关键点的坐标
                hand["box_size"] = box_size  # 方框大小参数
                hand["center"] = (x_center, y_center)  # 手的中心点
                all_hands.append(hand)  # 保存该只手的所有信息

                # 绘图
                if draw:
                    # 画方框
                    cv2.rectangle(
                        img,
                        (box_size[0] - 20, box_size[1] - 20),
                        (
                            box_size[0] + box_size[2] + 20,
                            box_size[1] + box_size[3] + 20,
                        ),
                        (255, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        hand["type"],
                        (box_size[0] - 30, box_size[1] - 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 255),
                        2,
                    )
                    # 关键点连线
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.landmark_style,
                        connection_drawing_spec=self.connect_style,
                    )

        return all_hands, img

    def fingers_up(self, hand):
        """检测手指垂直向上指的数量

        :param hand: 手的各种信息 (hand=hand_info[0])
        :return: 五个手指是否向上竖着 (0: 没有  1: 有)
        """

        fingers = []  # 保存手指是否向上的数据
        hand_lms_list = hand["lms_list"]  # 手的21个关键点坐标

        # 如果检测到手
        if self.results.multi_hand_landmarks:
            # 拇指向上判断
            if (
                hand_lms_list[self.tip_ids[0] - 1][0] - 15
                < hand_lms_list[self.tip_ids[0]][0]
                < hand_lms_list[self.tip_ids[0] - 1][0] + 15
                and hand_lms_list[self.tip_ids[0]][1]
                < hand_lms_list[self.tip_ids[0] - 1][1]
            ):
                fingers.append(1)
            else:
                fingers.append(0)

            # 其他四个手指向上判断
            for index in range(1, 5):
                if (
                    hand_lms_list[self.tip_ids[index] - 1][0] - 15
                    < hand_lms_list[self.tip_ids[index]][0]
                    < hand_lms_list[self.tip_ids[index] - 1][0] + 15
                    and hand_lms_list[self.tip_ids[index]][1]
                    < hand_lms_list[self.tip_ids[index] - 2][1]
                ):
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def fingers_straight(self, hand, flip=True):
        """检测手指伸直的数量（不弯曲的手指数）

        :param hand: 手的各种信息 (hand=hand_info[0])
        :param flip: 图像是否是翻转过的（这会影响拇指是否伸直的判断）
        :return: 五个手指是否伸直 (0: 没有  1: 有)
        """

        fingers = []  # 保存手指是否伸直的数据
        hand_lms_list = hand["lms_list"]  # 手的21个关键点坐标
        hand_type = hand["type"]  # 左手还是右手

        # 如果检测到手
        if self.results.multi_hand_landmarks:
            # 拇指伸直判断
            if flip is True:
                if hand_type == "Right":  # 如果是右手
                    if (
                        hand_lms_list[self.tip_ids[0]][0]
                        > hand_lms_list[self.tip_ids[0] - 1][0]
                    ):
                        fingers.append(0)
                    else:
                        fingers.append(1)
                else:  # 如果是左手
                    if (
                        hand_lms_list[self.tip_ids[0]][0]
                        > hand_lms_list[self.tip_ids[0] - 1][0]
                    ):
                        fingers.append(1)
                    else:
                        fingers.append(0)
            else:
                if hand_type == "Right":  # 如果是右手
                    if (
                        hand_lms_list[self.tip_ids[0]][0]
                        > hand_lms_list[self.tip_ids[0] - 1][0]
                    ):
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:  # 如果是左手
                    if (
                        hand_lms_list[self.tip_ids[0]][0]
                        > hand_lms_list[self.tip_ids[0] - 1][0]
                    ):
                        fingers.append(0)
                    else:
                        fingers.append(1)

            # 其他四个手指向上判断
            for index in range(1, 5):
                if (
                    hand_lms_list[self.tip_ids[index]][1]
                    < hand_lms_list[self.tip_ids[index] - 2][1]
                ):
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    @staticmethod
    def find_distance(p1, p2, img=None, draw=True):
        """计算两个点之间的距离

        :param p1: 第一个指定的点 (p1=hand_info[0]['lms_list'][p1])
        :param p2: 第二个指定的点 (p2=hand_info[0]['lms_list'][p2])
        :param img: 指定的图像
        :param draw: 是否进行绘画
        :return: length: 两点的距离
                 info: 两个点的(x1, y1, x2, y2, 中点x, 中点y)坐标
        """

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # 计算中点
        length = math.hypot(x2 - x1, y2 - y1)  # 计算两点的距离
        info = (x1, y1, x2, y2, cx, cy)
        if draw and img is not None:
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return length, info


# 主函数
def main():
    hand_detect = HandDetector(max_hands=1, detection_con=0.7)
    cap = cv2.VideoCapture(0)  # 打开摄像头
    cap.set(3, 640)  # 设置视频窗口长度
    cap.set(4, 480)  # 设置视频窗口宽度

    # 需要拖放的图像
    vir_img = cv2.imread("../resources/test.jpg")  # 导入需要拖放的图片文件
    vir_img = cv2.resize(vir_img, (vir_img.shape[1] // 3, vir_img.shape[0] // 3))
    vir_img_h, vir_img_w, vir_img_c = vir_img.shape  # 该图片的尺寸大小
    vir_img_min_x, vir_img_min_y = 0, 0  # 图像原始最小坐标点

    start_time, end_time = time.time(), 0  # 计时，用于计算帧数

    while True:
        success, img = cap.read()

        if not success:
            print("未检测到摄像头")
            break
        if cv2.waitKey(1) in [ord("q"), 27]:
            print("用户主动停止程序")
            break

        hand_info, img = hand_detect.find_hands(
            img=img, draw=True, flip=True
        )  # 寻找手是否存在
        if hand_info:  # 获取到手的位置信息
            finger = hand_detect.fingers_straight(hand=hand_info[0])
            # 只有食指和中指伸直的情况下
            if finger[0] == finger[3] == finger[4] == 0 and finger[1] == finger[2] == 1:
                length, info = hand_detect.find_distance(
                    p1=hand_info[0]["lms_list"][8],
                    p2=hand_info[0]["lms_list"][12],
                    img=img,
                    draw=True,
                )
                if length < 32:  # 食指和中指合并
                    if (
                        vir_img_min_x
                        < hand_info[0]["lms_list"][8][0]
                        < vir_img_min_x + vir_img_w
                        and vir_img_min_y
                        < hand_info[0]["lms_list"][8][1]
                        < vir_img_min_y + vir_img_h
                    ):
                        vir_img_min_x = hand_info[0]["lms_list"][8][0] - vir_img_w // 2
                        vir_img_min_y = hand_info[0]["lms_list"][8][1] - vir_img_h // 2

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

        img[
            vir_img_min_y : vir_img_min_y + vir_img_h,
            vir_img_min_x : vir_img_min_x + vir_img_w,
        ] = vir_img
        cv2.imshow("capture", img)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
