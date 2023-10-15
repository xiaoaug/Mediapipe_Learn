#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 10:27
# @File    : Mediapipe人脸检测.py
# @Software: PyCharm
# @Desc    :

import cv2
import mediapipe as mp  # 用于人脸追踪，pip install mediapipe
import time  # 用于计时，算出画面帧数


def image_face_detect(filepath):
    """图像人脸检测

    :param filepath: 图片文件路径
    """

    mp_faces = mp.solutions.face_detection  # 导入人脸检测模型
    faces = mp_faces.FaceDetection(0.5, 0)  # 内部参数分别为：置信度，选择模型（0：人脸距离镜头2m内模，1:5m远之外）

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出人脸位置
    keypoint_style = mp_draw.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=2
    )  # 关键点样式
    face_box_style = mp_draw.DrawingSpec(
        color=(255, 0, 0), thickness=2, circle_radius=2
    )  # 人脸预测框样式

    image = cv2.imread(filepath)  # 打开图片文件
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB
    faces_result = faces.process(image_rgb)  # 检测人脸位置

    if faces_result.detections:  # 如果检测到了人脸，则标注出来
        for detection in faces_result.detections:
            mp_draw.draw_detection(
                image,
                detection,
                keypoint_drawing_spec=keypoint_style,
                bbox_drawing_spec=face_box_style,
            )
        print("---------------- 人脸位置信息 ----------------")
        print(faces_result.detections[0])  # 查看第一个人脸的位置信息
        print("----------------- 人脸置信度 -----------------")
        print(faces_result.detections[0].score[0])
        print("---------------------------------------------")

    # 打印图片尺寸信息
    image_height, image_weight = image.shape[0], image.shape[1]
    print("图像高{}像素，宽{}像素".format(image_height, image_weight))

    cv2.imshow("image", image)  # 显示图片
    # cv2.imwrite('output/mediapipe_face_detect.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def camera_face_detect():
    """摄像头人脸检测"""

    mp_faces = mp.solutions.face_detection  # 导入人脸检测模型
    faces = mp_faces.FaceDetection()  # 内部参数分别为：置信度，选择模型（0：人脸距离镜头2m内模，1:5m远之外）

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出人脸位置
    keypoint_style = mp_draw.DrawingSpec(
        color=(0, 0, 255), thickness=2, circle_radius=3
    )  # 关键点样式
    face_box_style = mp_draw.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=3
    )  # 人脸预测框样式

    cap = cv2.VideoCapture(0)  # 打开摄像头
    start_time, end_time = time.time(), 0

    while True:
        flag, frame = cap.read()
        if flag:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_result = faces.process(frame_rgb)  # 检测人脸位置
            # print(faces_result.detections)

            if faces_result.detections:  # 如果检测到了人脸，则画出来
                for detection in faces_result.detections:
                    mp_draw.draw_detection(
                        frame,
                        detection,
                        keypoint_drawing_spec=keypoint_style,
                        bbox_drawing_spec=face_box_style,
                    )
            else:  # 未检测到人脸
                cv2.putText(
                    frame,
                    "No Face Detected",
                    (25, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

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


def video_face_detect(filepath):
    """视频人脸检测，并生成结果文件

    :param filepath: 视频文件路径
    """

    mp_faces = mp.solutions.face_detection  # 导入人脸检测模型
    faces = mp_faces.FaceDetection(0.5, 0)  # 内部参数分别为：置信度，选择模型（0：人脸距离镜头2m内模，1:5m远之外）

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出人脸位置
    keypoint_style = mp_draw.DrawingSpec(
        color=(0, 0, 255), thickness=2, circle_radius=3
    )  # 关键点样式
    face_box_style = mp_draw.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=3
    )  # 人脸预测框样式

    video = cv2.VideoCapture(filepath)  # 打开视频文件

    # 用于保存处理好的视频文件
    frame_size = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )  # 获取当前帧大小
    writer = cv2.VideoWriter(
        filename="../../ZhangXiao_Project/Opencv/output/mediapipe_face_detect.mp4",
        fourcc=cv2.VideoWriter_fourcc(*"MP4V"),  # 视频编码格式
        fps=video.get(cv2.CAP_PROP_FPS),  # 视频帧率
        frameSize=(frame_size[0], frame_size[1]),
    )  # 视频宽高

    while True:
        flag, frame = video.read()
        if not flag:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
        faces_result = faces.process(frame_rgb)  # 检测人脸位置
        print(faces_result.detections)

        if faces_result.detections:  # 如果检测到了人脸，则画出来
            for detection in faces_result.detections:
                mp_draw.draw_detection(
                    frame,
                    detection,
                    keypoint_drawing_spec=keypoint_style,
                    bbox_drawing_spec=face_box_style,
                )
        else:  # 未检测到人脸
            cv2.putText(
                frame,
                "No Face Detected",
                (25, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("image", frame)  # 显示摄像头画面
        writer.write(frame)

        if cv2.waitKey(1) in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()
    video.release()
    writer.release()
    print("视频文件已生成！")


if __name__ == "__main__":
    camera_face_detect()
