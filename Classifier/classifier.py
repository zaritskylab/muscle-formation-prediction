import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns


# import Scripts.DataPreprocessing.CellsCropper as CellsCropper


def metrics(true_y, pred_y):
    print("Accuracy:", accuracy_score(true_y, pred_y))
    print("Precision:", precision_score(true_y, pred_y))
    print("Recall:", recall_score(true_y, pred_y))


def plot_svm_pred_vs_realtime(data, svm_classifier, title):
    real_time = data["time"]
    full_video_data = data.drop("time", 1).astype('int')
    svm_pred_prob = svm_classifier.predict_proba(full_video_data)[:, 1]
    svm_pred = svm_classifier.predict(full_video_data)

    columns = ["real_time", "svm_pred_prob", "svm_pred"]
    df = pd.DataFrame(columns=columns)
    df["real_time"] = real_time
    df["svm_pred_prob"] = svm_pred_prob
    df["svm_pred"] = svm_pred

    sns.set_style("whitegrid")
    plt.suptitle(title)
    sns.scatterplot(data=df, x="real_time", y="svm_pred_prob", hue="real_time")
    plt.suptitle(title)
    # sns.displot(data, x="time")
    plt.show()


if __name__ == '__main__':
    random.seed(10)

    # create_data(encoder_name ="encoder gen_learning_rate _0.0003_")

    train = pd.read_pickle("../DataPreprocessing/data/encoded_labeled - train normalized")
    test = pd.read_pickle("../DataPreprocessing/data/encoded_labeled - test normalized")
    full_timestamp_test = pd.read_pickle("../DataPreprocessing/data/test - encoded video 5 with timestamp")
    full_timestamp_control = pd.read_pickle("../DataPreprocessing/data/test - encoded video 7 with timestamp")
    single_cell_diff = pd.read_pickle("../DataPreprocessing/data/single cell - exp 5 - diff")
    single_cell_con = pd.read_pickle("../DataPreprocessing/data/single cell - exp 7 - con")

    print("train set size: {}".format(len(train)))
    print("video4_test set size: {}".format(len(test)))
    print("number of images in exp 5: {}".format(len(full_timestamp_test)))
    print("number of images in exp 7: {}".format(len(full_timestamp_control)))
    print("number of images in single cell- exp 5: {}".format(len(single_cell_diff)))
    print("number of images in single cell- exp 7: {}".format(len(single_cell_con)))

    X_train = train.drop("time", 1).astype('int')
    y_train = train["time"].astype('int')

    X_test = test.drop("time", 1).astype('int')
    y_test = test["time"].astype('int')

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    importance = pd.Series(clf.feature_importances_)
    importance.sort_values(ascending=False)
    print(importance)
    imp_frame = importance.to_frame()
    imp_frame.plot(kind="bar")
    plt.xticks([])
    plt.xlabel('Features')
    plt.ylabel('Importances')
    plt.title('Feature Importance Plot')
    plt.show()

    predicted = cross_val_predict(clf, X_test, y_test, cv=10)
    print(classification_report(y_test, predicted))
    metrics(y_test, predicted)

    plot_svm_pred_vs_realtime(data=test, svm_classifier=clf, title="video4_test data")
    # plot_svm_pred_vs_realtime(data=full_timestamp_test, svm_classifier=clf, title="full exp- 5")
    # plot_svm_pred_vs_realtime(data=full_timestamp_control, svm_classifier=clf, title="full exp- 7")
    plot_svm_pred_vs_realtime(data=single_cell_diff, svm_classifier=clf, title="single diff cell")
    plot_svm_pred_vs_realtime(data=single_cell_con, svm_classifier=clf, title="single con cell")
