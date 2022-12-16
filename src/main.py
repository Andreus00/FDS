from sklearn.svm import SVC
from sklearn.svm import SVR

from load_data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.naive_bayes import GaussianNB
import numpy as np
import tqdm
import config
import matplotlib.pyplot as plt

if __name__ == "__main__":
    CLASSIFY = False
    REGRESS = True
    vid = 1
    # classifiers
    models = [(str(model), model(), model(), model()) for model in [DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, LogisticRegression, GaussianNB]]
    regressors = [(str(model), model()) for model in [LinearRegression]]
    # get dataset

    d = DataLoader()
    d.read_dataset()
    d.shuffle_videos()
    if CLASSIFY:
        X, y, images = d.sample_dataset(video=vid, return_images=True)
        X_train, X_test, y_train, y_test, im_train, im_test = train_test_split(X, y, images, stratify=y)
        X_skel = X_train[:, 0:9]
        X_clothes = X_train[:, 9:9 + config.NUM_POINTS_LBP]
        X_face = X_train[:, 9 + config.NUM_POINTS_LBP:]
        X_skel_test = X_test[:, 0:9]
        X_clothes_test = X_test[:, 9:9 + config.NUM_POINTS_LBP]
        X_face_test = X_test[:, 9 + config.NUM_POINTS_LBP:]
        y = np.array(y_train).flatten()

        # fit  and predict models
        for name, model_skel, model_clothes, model_face in models:
            model_skel.fit(X_skel, y_train)
            model_clothes.fit(X_clothes, y_train)
            model_face.fit(X_face, y_train)

            print("-" * 10)
            print("|", name, "|")
            # Test
            print("-" * 10)
            skel_pred = model_skel.predict(X_skel_test)
            print(classification_report(y_test, skel_pred))
            print("-" * 10)
            clothes_pred = model_clothes.predict(X_clothes_test)
            print(classification_report(y_test, clothes_pred))
            print("-" * 10)
            face_pred = model_face.predict(X_face_test)
            print(classification_report(y_test, face_pred))
        

    # Regression

            print("-" * 20)
            print("-" * 20)
            print("-" * 20)

    if REGRESS:
        X, y, images = d.sample_dataset(video=vid, return_images=True)
        X_logistic, y_logistc = d.sample_dataset(video=vid, return_images=False)
        X_train, X_test, y_train, y_test, im_train, im_test = train_test_split(X, y, images, stratify=y)
        # X_logistic, X_test, y_logistc, y_test = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5)
        
        for name, model in regressors:
            model.fit(X_train, y_train)

            # Train the logistic regressor
            log_reg = LogisticRegression()
            log_reg.fit(model.predict(X_logistic).reshape(-1, 1), y_logistc)
            
            
            # Test
            print("-" * 10)
            print("|", name, "|")
            print("-" * 10)
            pred = model.predict(X_test)
            log_pred = log_reg.predict(pred.reshape(-1, 1))

            print(classification_report(y_test, log_pred))
            print(mean_absolute_error(y_test, pred))
            print(mean_squared_error(y_test, pred))
            # for a, b in zip(y_test, pred):
            #     print(a, b)

            # for idx, res in enumerate(zip(y_test, zip(skel_pred, clothes_pred, face_pred))):
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(im_test[idx])
            #     ax[0].set_title("True label: " + str(bool(res[0])))
            #     ax[1].imshow(im_train[1])
            #     ax[1].set_title(str(res[1]))#"skel label: " + str(bool(res[1][0])), "clothes label: " + str(bool(res[1][1])), "face label: " + str(bool(res[1][2])))
            #     print(res)
            #     plt.show()

