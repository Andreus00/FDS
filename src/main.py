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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import tqdm
import config
import math
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    CLASSIFY = True
    REGRESS = False
    GENERATE_DATASET = False
    vid = 1
    # classifiers
    models = [(str(model), model(), model(), model()) for model in [DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, LogisticRegression, GaussianNB, SVC]]
    regressors = [(str(model), model) for model in [LinearRegression, Pipeline, SVR, SVR, SVR]]
    # get dataset

    d = DataLoader()
    d.read_dataset()
    d.shuffle_videos()
    if CLASSIFY:
        # X, y, images = d.sample_dataset(video=vid, return_images=True)
        X = np.load(config.SAMPLED_PATH + "sample_0_X.npy")
        y = np.load(config.SAMPLED_PATH + "sample_0_y.npy")
        images = []
        X_skel, X_clothes, X_face, X_skel_test, X_clothes_test, X_face_test, y_train, y_test, y_face, y_face_test = filter_and_split_dataset(X, y)

        # fit  and predict models
        for name, model_skel, model_clothes, model_face in models:
            if model_skel is LogisticRegression:
                model_skel.fit(X_skel, y_train, max_iter=10000)
                model_clothes.fit(X_clothes, y_train, max_iter=10000)
                model_face.fit(X_face, y_face, max_iter=10000)
            else:
                model_skel.fit(X_skel, y_train)
                model_clothes.fit(X_clothes, y_train)
                model_face.fit(X_face, y_face)


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
            print(classification_report(y_face_test, face_pred))
        

    # Regression

            print("-" * 20)
            print("-" * 20)
            print("-" * 20)

    if REGRESS:
        X, y, images = d.sample_dataset(video=vid, return_images=True)
        y -= 1
        y *= -1
        X_logistic, y_logistc, _ = d.sample_dataset(video=vid, return_images=False)
        X_skel, X_clothes, X_face, X_skel_test, X_clothes_test, X_face_test, y_train, y_test, y_face, y_face_test = filter_and_split_dataset(X, y)
        X_skel_logistic, X_clothes_logistic, X_face_logistic, X_skel_test_logistic, X_clothes_test_logistic, X_face_test_logistic, y_train_logistic, y_test_logistic, y_face_logistic, y_face_test_logistic = filter_and_split_dataset(X_logistic, y_logistc)
        c_svr = 0
        params_smr = [(2, "poly"), (3, "poly"), (2, "linear")]
        for name, m in regressors:
            model_skel = model_clothes = model_face = None
            if m is Pipeline:
                model_skel, model_clothes, model_face = m([("poly", PolynomialFeatures()), ("Regression", LinearRegression())]), m([("poly", PolynomialFeatures()), ("Regression", LinearRegression())]), m([("poly", PolynomialFeatures()), ("Regression", LinearRegression())])
            elif m is SVR:
                model_skel, model_clothes, model_face = m(degree=params_smr[c_svr][0], kernel=params_smr[c_svr][1]), m(degree=params_smr[c_svr][0], kernel=params_smr[c_svr][1]), m(degree=params_smr[c_svr][0], kernel=params_smr[c_svr][1])
                c_svr += 1
            else:
                model_skel, model_clothes, model_face = m(), m(), m()

            if m is LogisticRegression:
                model_skel.fit(X_skel, y_train, max_iter=10000)
                model_clothes.fit(X_clothes, y_train, max_iter=10000)
                model_face.fit(X_face, y_train, max_iter=10000)
            else:
                model_skel.fit(X_skel, y_train)
                model_clothes.fit(X_clothes, y_train)
                model_face.fit(X_face, y_face)

            # Train the logistic regressor
            log_reg_skel, log_reg_clothes, log_reg_face = LogisticRegression(), LogisticRegression(), LogisticRegression()
            log_reg_skel.fit(model_skel.predict(X_skel_logistic).reshape(-1, 1), y_train_logistic)
            log_reg_clothes.fit(model_clothes.predict(X_clothes_logistic).reshape(-1, 1), y_train_logistic)
            log_reg_face.fit(model_face.predict(X_face_logistic).reshape(-1, 1), y_face_logistic)
            
            
            # Test
            print("-" * 10)
            print("|", name, "|")
            print("-" * 10)
            print("| skel |")
            skel_pred = model_skel.predict(X_skel_test)
            log_skel_pred = log_reg_skel.predict(skel_pred.reshape(-1, 1))

            print(classification_report(y_test, log_skel_pred))
            print(mean_absolute_error(y_test, skel_pred))
            print(mean_squared_error(y_test, skel_pred))


            print("| clothes |")
            clotehs_pred = model_clothes.predict(X_clothes_test)
            log_clotehs_pred = log_reg_clothes.predict(clotehs_pred.reshape(-1, 1))

            print(classification_report(y_test, log_clotehs_pred))
            print(mean_absolute_error(y_test, clotehs_pred))
            print(mean_squared_error(y_test, clotehs_pred))


            print("| face |")
            face_pred = model_face.predict(X_face_test)
            log_face_pred = log_reg_face.predict(face_pred.reshape(-1, 1))

            print(classification_report(y_face_test, log_face_pred))
            print(mean_absolute_error(y_face_test, face_pred))
            print(mean_squared_error(y_face_test, face_pred))

            plt.plot(y_test, clotehs_pred, 'o')
            plt.show()

            plt.plot(y_test, skel_pred, 'o')
            plt.show()

            # for idx, res in enumerate(zip(y_test, zip(skel_pred, clothes_pred, face_pred))):
            #     fig, ax = plt.subplots(1, 2)
            #     ax[0].imshow(im_test[idx])
            #     ax[0].set_title("True label: " + str(bool(res[0])))
            #     ax[1].imshow(im_train[1])
            #     ax[1].set_title(str(res[1]))#"skel label: " + str(bool(res[1][0])), "clothes label: " + str(bool(res[1][1])), "face label: " + str(bool(res[1][2])))
            #     print(res)
            #     plt.show()
    
    if GENERATE_DATASET:
        for i in range(0, 50):
            d.sample_dataset(video=vid, return_images=True, file="sample_"+str(i))



