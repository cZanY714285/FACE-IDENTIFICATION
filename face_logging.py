import cv2
import dlib
import numpy as np
import os
import pickle
# 初始化人脸检测和特征提取模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('D:/mess/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('D:/mess/dlib_face_recognition_resnet_model_v1.dat')
# 保存录入的人脸数据
face_db = {}  # 用于存储人脸特征和对应名称的字典
def save_face_data(name, face_descriptor):
    # 将人脸特征保存到数据库
    face_db[name] = face_descriptor
    with open('face_data.pkl', 'wb') as f:
        pickle.dump(face_db, f)
def load_face_data():
    # 加载已保存的人脸数据
    if os.path.exists('face_data.pkl'):
        with open('face_data.pkl', 'rb') as f:
            return pickle.load(f)
    return {}
# 加载已经录入的面部数据
face_db = load_face_data()
# 从摄像头捕捉人脸并录入
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法捕获摄像头画面")
        break
    # 人脸检测
    faces = detector(frame, 1)
    # 绘制人脸矩形框并处理
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        # 特征提取
        shape = sp(frame, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        # 转换人脸特征为可比对的格式 (list -> tuple)
        face_descriptor_tuple = tuple(face_descriptor)
        # 检查是否已录入
        match_found = False
        for name, saved_descriptor in face_db.items():
            if np.linalg.norm(np.array(saved_descriptor) - np.array(face_descriptor_tuple)) < 0.6:
                print(f"已识别为 {name}")
                match_found = True
                break
        # 如果没有找到匹配的面部特征，要求用户输入名字
        if not match_found:
            name = input("检测到新的人脸，请输入名字：")
            save_face_data(name, face_descriptor_tuple)
    # 显示视频
    cv2.imshow('Camera', frame)
    # 按下 '8' 键退出
    if cv2.waitKey(1) & 0xFF == ord('8'):
        print("退出程序")
        break
# 释放资源
cap.release()
cv2.destroyAllWindows()
