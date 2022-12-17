from sklearn.model_selection import train_test_split
import config
import math
import numpy as np

def filter_and_split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    X_skel = X_train[:, 0:9]
    X_clothes = X_train[:, 9:9 + config.NUM_POINTS_LBP]
    X_face_not_filtered = X_train[:, 9 + config.NUM_POINTS_LBP:]
    X_face = []
    y_face = []
    for X_f, y_f in zip(X_face_not_filtered, y_train):
        if not math.isnan(X_f[0]):
            X_face.append(X_f)
            y_face.append(y_f)

    X_face = np.asarray(X_face)
    y_face = np.asarray(y_face).flatten()
    
    X_skel_test = X_test[:, 0:9]
    X_clothes_test = X_test[:, 9:9 + config.NUM_POINTS_LBP]
    X_face_test_not_filtered = X_test[:, 9 + config.NUM_POINTS_LBP:]


    X_face_test = []
    y_face_test = []
    for X_f, y_f in zip(X_face_test_not_filtered, y_test):
        if not math.isnan(X_f[0]):
            X_face_test.append(X_f)
            y_face_test.append(y_f)
    
    X_face_test = np.asarray(X_face_test)
    y_face_test = np.asarray(y_face_test).flatten()

    y = np.array(y_train).flatten()

    return X_skel, X_clothes, X_face, X_skel_test, X_clothes_test, X_face_test, y_train, y_test, y_face, y_face_test
