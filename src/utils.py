from sklearn.model_selection import train_test_split
import config
import math
import numpy as np
from mediapipe.python.solutions import pose as mp_pose
import cv2

def filter_faces(X, y):
    X_face = []
    y_face = []
    for X_f, y_f in zip(X, y):
        if not math.isnan(X_f[0]):
            X_face.append(X_f)
            y_face.append(y_f)

    return np.asarray(X_face), np.asarray(y_face).flatten()


def filter_and_split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    X_skel = X_train[:, 0:9]
    X_clothes = X_train[:, 9:9 + config.NUM_POINTS_LBP]
    X_face_not_filtered = X_train[:, 9 + config.NUM_POINTS_LBP:]
    X_face, y_face = filter_faces(X_face_not_filtered, y_train)
    
    X_skel_test = X_test[:, 0:9]
    X_clothes_test = X_test[:, 9:9 + config.NUM_POINTS_LBP]
    X_face_test_not_filtered = X_test[:, 9 + config.NUM_POINTS_LBP:]

    X_face_test, y_face_test = filter_faces(X_face_test_not_filtered, y_test)

    y = np.array(y_train).flatten()

    return X_skel, X_clothes, X_face, X_skel_test, X_clothes_test, X_face_test, y_train, y_test, y_face, y_face_test


def get_pose(img):
    ret = None
    success = False
    with mp_pose.Pose() as pose_tracker:
        result = pose_tracker.process(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pose_landmarks = result.pose_landmarks
        if pose_landmarks is not None:
            ret = pose_landmarks
            success = True
    return ret, success

        
        
