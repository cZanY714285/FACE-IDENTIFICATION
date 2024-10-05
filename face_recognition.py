import cv2
import dlib
import numpy as np
import pickle
# 加载已保存的人脸数据
with open('face_data.pkl', 'rb') as f:
    face_db = pickle.load(f)
# 初始化人脸检测和特征提取模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('D:/mess/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('D:/mess/dlib_face_recognition_resnet_model_v1.dat')
# 计算欧氏距离
def euclidean_distance(face_descriptor1, face_descriptor2):
    return np.linalg.norm(np.array(face_descriptor1) - np.array(face_descriptor2))
def recognize_face(face_descriptor, threshold=0.6):
    # 在已录入的人脸数据库中查找匹配
    min_distance = float('inf')
    recognized_name = None
    for name, saved_descriptor in face_db.items():
        distance = euclidean_distance(face_descriptor, saved_descriptor)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            recognized_name = name
    return recognized_name
# 摄像头
cap = cv2.VideoCapture(0)
# 检查摄像头是否成功打开
if not cap.isOpened():
    print("未打开摄像头")
    exit()
paused = False
def face_detect_demo(frame):
    # 将彩色图像转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用dlib的cnn人脸检测器检测人脸
    faces = detector(gray_frame, 1)
    if len(faces) == 0:
        # 如果没有检测到人脸，显示错误信息
        print("未检测到人脸")
    else:
        print(f"检测到 {len(faces)} 张人脸")
        for face in faces:
            # Dlib返回的是dlib.rectangles对象，因此要获取每个人脸的坐标
            x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 尝试特征提取并捕获可能的错误
            try:
                shape = sp(gray_frame, face)
                face_descriptor = facerec.compute_face_descriptor(frame, shape)

                # 识别人脸
                recognized_name = recognize_face(face_descriptor)
                if recognized_name:
                    cv2.putText(frame, f"recoginized as: {recognized_name}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "stranger", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            except Exception as e:
                print(f"特征提取失败: {e}")
# 实时处理视频流
while True:
    if not paused:
        # 读取摄像头帧
        ret, frame = cap.read()
        # 检查是否成功读取帧
        if not ret:
            print("无法接收帧 (摄像头已断开)")
            break
        # 调用人脸检测函数
        face_detect_demo(frame)
        # 显示结果
        cv2.imshow('Camera', frame)
    # 检测按键
    key = cv2.waitKey(1) & 0xFF
    # 按下 '9' 时暂停/恢复
    if key == ord('9'):
        paused = not paused
    # 按下 '7' 时退出
    if key == ord('7'):
        print('实时检测结束')
        break
# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()




