#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 22:30
# @File    : Mediapipe手部追踪.py
# @Software: PyCharm
# @Desc    : 手部追踪功能

import cv2
import mediapipe as mp  # 用于手的追踪，pip install mediapipe
import time  # 用于计时，算出画面帧数


def camera_hand_track():
    """摄像头手部追踪"""

    mp_hands = mp.solutions.hands  # 引入手部模型
    hands = mp_hands.Hands(
        False, 2, 1, 0.7, 0.5
    )  # 内部参数分别为：是否静态图片模式，检测手的最大数量，模型复杂度，最低检测置信度，最低跟踪置信度

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出手部位置
    hand_landmark_style = mp_draw.DrawingSpec(
        color=(0, 0, 255), thickness=4
    )  # 设定画出手位置的颜色参数
    hand_connect_style = mp_draw.DrawingSpec(
        color=(0, 255, 0), thickness=8
    )  # 设定画出连线的颜色参数

    cap = cv2.VideoCapture(0)  # 打开摄像头
    start_time, end_time = time.time(), 0

    while True:
        flag, frame = cap.read()
        if flag:
            frame = cv2.flip(frame, 1)  # 图像镜像反转
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_result = hands.process(frame_rgb)  # 检测手部位置
            # print(hands_result.multi_hand_landmarks)   # 显示手的坐标
            if hands_result.multi_hand_landmarks:  # 如果检测到手的位置，则画出来
                for hand_landmark in hands_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmark,
                        mp_hands.HAND_CONNECTIONS,
                        hand_landmark_style,
                        hand_connect_style,
                    )
                    # 获取手位置的21个点的所有坐标值
                    for index, landmark in enumerate(hand_landmark.landmark):
                        hand_x_position = int(
                            landmark.x * frame.shape[1]
                        )  # 手部X坐标，加int是为了下面的putText不出错
                        hand_y_position = int(landmark.y * frame.shape[0])  # 手部Y坐标

                        # 在每一帧中标出每个点的序号
                        cv2.putText(
                            frame,
                            str(index),
                            (hand_x_position - 25, hand_y_position + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            2,
                        )
                        print(index, hand_x_position, hand_y_position)

        # 计算图像帧数
        end_time = time.time()  # 获取当前时间
        fps = 1 // (end_time - start_time)  # 计算帧数
        start_time = end_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

        cv2.imshow("image", frame)  # 显示摄像头画面

        if cv2.waitKey(1) in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()
    cap.release()


def camera_hand_track_colorful():
    """摄像头手部追踪（多彩版）"""

    mp_hands = mp.solutions.hands  # 引入手部模型
    hands = mp_hands.Hands(
        False, 2, 1, 0.7, 0.5
    )  # 内部参数分别为：是否静态图片模式，检测手的最大数量，模型复杂度，最低检测置信度，最低跟踪置信度

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出手部位置
    hand_connect_style = mp_draw.DrawingSpec(
        color=(0, 255, 0), thickness=2
    )  # 设定画出连线的颜色参数

    cap = cv2.VideoCapture(0)  # 打开摄像头
    start_time, end_time = time.time(), 0

    while True:
        flag, frame = cap.read()

        if not flag:  # 没有检测到视频，则退出
            print("未检测到视频文件")
            break

        frame = cv2.flip(frame, 1)  # 图像镜像反转
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
        hands_result = hands.process(frame_rgb)  # 检测手部位置
        hand_landmarks = hands_result.multi_hand_landmarks  # 手位置信息

        if hand_landmarks:  # 如果检测到手的位置
            handedness_info = ""  # 用于在图像中显示左手和右手数据
            for hand_index in range(len(hand_landmarks)):  # 检索每一只识别出来的手
                hand_21 = hand_landmarks[hand_index]  # 获取该只手21个关键点坐标
                mp_draw.draw_landmarks(
                    frame,
                    hand_21,
                    mp_hands.HAND_CONNECTIONS,
                    connection_drawing_spec=hand_connect_style,
                )  # 可视化关键点连接

                # 记录左右手的信息
                handedness_label = (
                    hands_result.multi_handedness[hand_index].classification[0].label
                )  # 读取该只手是左手还是右手
                handedness_score = (
                    hands_result.multi_handedness[hand_index].classification[0].score
                )  # 手的置信度
                handedness_info += (
                    f"{hand_index}:{handedness_label}:{handedness_score:.2f}  "  # 用于显示
                )

                # 给该只手的关键点进行自定义染色
                hand_z_0 = hand_21.landmark[0].z  # 获取手腕根部深度坐标，根据它来控制手的关键点的圆点大小
                for i in range(21):  # 遍历该只手的21个关键点
                    # 获取3D坐标
                    hand_x = int(hand_21.landmark[i].x * frame.shape[1])
                    hand_y = int(hand_21.landmark[i].y * frame.shape[0])
                    hand_z = hand_21.landmark[i].z
                    depth_z = hand_z_0 - hand_z  # 以手腕根部为基准，得到该关键点的深度差值
                    radius = max(int(6 + 40 * depth_z), 0)  # 用圆的半径反映深度大小

                    # 自定义染色
                    if i == 0:  # 手腕
                        cv2.circle(
                            frame, (hand_x, hand_y), radius, (0, 0, 255), -1
                        )  # -1表示填充
                    if i in [1, 5, 9, 13, 17]:  # 指根
                        cv2.circle(frame, (hand_x, hand_y), radius, (16, 144, 247), -1)
                    if i in [2, 6, 10, 14, 18]:  # 第一指节
                        cv2.circle(frame, (hand_x, hand_y), radius, (1, 240, 255), -1)
                    if i in [3, 7, 11, 15, 19]:  # 第二指节
                        cv2.circle(frame, (hand_x, hand_y), radius, (140, 47, 240), -1)
                    if i in [4, 8, 12, 16, 20]:  # 指尖
                        cv2.circle(frame, (hand_x, hand_y), radius, (223, 155, 60), -1)

                    # 标出每个关键点的序号
                    cv2.putText(
                        frame,
                        str(i),
                        (hand_x - 25, hand_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        2,
                    )

            # 在图像中写出左右手信息
            cv2.putText(
                frame,
                handedness_info,
                (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        # 计算图像帧数
        end_time = time.time()  # 获取当前时间
        fps = 1 // (end_time - start_time)  # 计算帧数
        start_time = end_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (0, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("image", frame)  # 显示摄像头画面

        if cv2.waitKey(1) in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    camera_hand_track_colorful()
