import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, recall_score, precision_score
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
    sns.scatterplot(data=df, x="real_time", y="svm_pred_prob", hue="svm_pred")
    plt.suptitle(title)
    # sns.displot(data, x="time")
    plt.show()


random.seed(10)

# create_data(encoder_name ="encoder gen_learning_rate _0.0003_")

train = pd.read_pickle("../DataPreprocessing/data/encoded_labeled - train normalized")
test = pd.read_pickle("../DataPreprocessing/data/encoded_labeled - test normalized")
full_timestamp_test = pd.read_pickle("../DataPreprocessing/test - encoded video 5 with timestamp")
full_timestamp_control = pd.read_pickle("../../Scripts/DataPreprocessing/test - encoded video 7 with timestamp")
single_cell_diff = pd.read_pickle("../../Scripts/DataPreprocessing/single cell - exp 5 - diff")
single_cell_con = pd.read_pickle("../../Scripts/DataPreprocessing/single cell - exp 7 - con")

print("train set size: {}".format(len(train)))
print("test set size: {}".format(len(test)))
print("number of images in exp 5: {}".format(len(full_timestamp_test)))
print("number of images in exp 7: {}".format(len(full_timestamp_control)))
print("number of images in single cell- exp 5: {}".format(len(single_cell_diff)))
print("number of images in single cell- exp 7: {}".format(len(single_cell_con)))

X_train = train.drop("time", 1).astype('int')
y_train = train["time"].astype('int')

X_test = test.drop("time", 1).astype('int')
y_test = test["time"].astype('int')

svm_classifier = svm.SVC(probability=True, C=1000)#svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)

# svm prediction- test set
y_pred_svm = svm_classifier.predict(X_test)
print(svm_classifier.predict_proba(X_test))
print("svm netrics:")
metrics(y_test, y_pred_svm)

plot_svm_pred_vs_realtime(data=test, svm_classifier=svm_classifier, title="test data")
plot_svm_pred_vs_realtime(data=full_timestamp_test, svm_classifier=svm_classifier, title="full exp- 5")
plot_svm_pred_vs_realtime(data=full_timestamp_control, svm_classifier=svm_classifier, title="full exp- 7")
plot_svm_pred_vs_realtime(data=single_cell_diff, svm_classifier=svm_classifier, title="single diff cell")
plot_svm_pred_vs_realtime(data=single_cell_con, svm_classifier=svm_classifier, title="single con cell")

# naive_base_classifier = GaussianNB()
# naive_base_classifier.fit(X_train, y_train)
# y_pred_svm = naive_base_classifier.predict(X_test)
# probs_nb = naive_base_classifier.predict_proba(X_test)
