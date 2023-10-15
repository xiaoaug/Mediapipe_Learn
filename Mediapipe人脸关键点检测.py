#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 17:11
# @File    : Mediapipe人脸关键点检测.py
# @Software: PyCharm
# @Desc    :

import cv2
import mediapipe as mp
import time


def image_face_keypoint_detect(filepath):
    """图片人脸关键点检测

    :param filepath: 图片文件路径
    """

    mp_faces = mp.solutions.face_mesh  # 导入人脸关键点检测模型
    faces = mp_faces.FaceMesh(True, 1, True)  # 参数分别为：图片还是视频、最多检测人脸数、是否精细定位、置信度阈值、追踪阈值

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出人脸关键点位置
    mp_draw_style = mp.solutions.drawing_styles  # 引入可视化样式

    image = cv2.imread(filepath)  # 打开图片文件
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB
    faces_result = faces.process(image_rgb)  # 检测人脸关键点位置
    face_landmarks = faces_result.multi_face_landmarks  # 人脸标记点数据

    if face_landmarks:  # 如果检测到了人脸，则标注出来
        print(f"检测到了{len(face_landmarks)}张脸")

        # 遍历每一张脸
        for face_landmark in face_landmarks:
            # 绘制人脸网格
            mp_draw.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_faces.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(66, 77, 229), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=mp_draw_style.get_default_face_mesh_tesselation_style(),
            )
            # 绘制脸轮廓、眼睫毛、眼眶、嘴唇
            mp_draw.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_faces.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(66, 77, 229), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=mp_draw_style.get_default_face_mesh_contours_style(),
            )
            # 绘制瞳孔区域
            mp_draw.draw_landmarks(
                image=image,
                landmark_list=face_landmark,
                connections=mp_faces.FACEMESH_IRISES,
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(10, 169, 77), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=mp_draw_style.get_default_face_mesh_iris_connections_style(),
            )
    else:
        print("未检测到人脸信息")

    # 打印图片尺寸信息
    image_height, image_weight = image.shape[0], image.shape[1]
    print("图像高{}像素，宽{}像素".format(image_height, image_weight))

    cv2.imshow("image", image)  # 显示图像
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def camera_face_keypoint_detect():
    """摄像头人脸关键点检测"""

    mp_faces = mp.solutions.face_mesh  # 导入人脸关键点检测模型
    faces = mp_faces.FaceMesh(False, 2, True)  # 参数分别为：图片还是视频、最多检测人脸数、是否精细定位、置信度阈值、追踪阈值

    mp_draw = mp.solutions.drawing_utils  # 引入绘画，用于画出人脸关键点位置
    mp_draw_style = mp.solutions.drawing_styles  # 引入可视化样式

    cap = cv2.VideoCapture(0)  # 打开摄像头
    start_time, end_time = time.time(), 0

    while True:
        flag, frame = cap.read()

        if not flag:
            print("未检测到摄像头")
            break

        frame = cv2.flip(frame, 1)  # 图像镜像反转
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
        faces_result = faces.process(frame_rgb)  # 检测人脸关键点位置
        face_landmarks = faces_result.multi_face_landmarks

        if face_landmarks:  # 如果检测到了人脸，则标注出来
            cv2.putText(
                frame,
                f"Face Num: {len(face_landmarks)}",
                (0, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # 遍历每一张脸
            for face_landmark in face_landmarks:
                # 绘制人脸网格
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=mp_faces.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(66, 77, 229), thickness=2, circle_radius=1
                    ),
                    connection_drawing_spec=mp_draw_style.get_default_face_mesh_tesselation_style(),
                )
                # 绘制脸轮廓、眼睫毛、眼眶、嘴唇
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=mp_faces.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(66, 77, 229), thickness=1, circle_radius=1
                    ),
                    connection_drawing_spec=mp_draw_style.get_default_face_mesh_contours_style(),
                )
                # 绘制瞳孔区域
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmark,
                    connections=mp_faces.FACEMESH_IRISES,
                    landmark_drawing_spec=mp_draw.DrawingSpec(
                        color=(10, 169, 77), thickness=1, circle_radius=1
                    ),
                    connection_drawing_spec=mp_draw_style.get_default_face_mesh_iris_connections_style(),
                )
        else:
            cv2.putText(
                frame,
                "No Face Detected",
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

        cv2.imshow("camera", frame)  # 显示摄像头画面

        if cv2.waitKey(1) in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    camera_face_keypoint_detect()
