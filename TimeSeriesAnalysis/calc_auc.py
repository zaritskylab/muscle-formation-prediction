from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from TimeSeriesAnalysis.ts_fresh import load_data


def calc_tpr_fpr(y_actual, y_hat, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_hat)):
        if y_hat[i] >= threshold:
            if y_actual[i] == True:
                tp += 1
            else:
                fp += 1
        elif y_hat[i] < threshold:
            if y_actual[i] == False:
                tn += 1
            else:
                fn += 1

    # Find True positive rate and False positive rate based on the threshold
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)

    return [fpr, tpr]


def plot_roc(fpr, tpr, auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


dir_name = "tmp_motility-True_intensity-False/140,170 frames ERK, [[0, 30]] frames con"

# load the model & train set & test set
clf, X_train, X_test, y_train, y_test = load_data(dir_name)
probs = clf.predict_proba(X_test)[:, 1]
thresholds = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1]

roc_points = []
for threshold in thresholds:
    rates = calc_tpr_fpr(np.array(y_test), probs, threshold)
    roc_points.append(rates)

fpr_array = []
tpr_array = []
for i in range(len(roc_points) - 1):
    point1 = roc_points[i];
    point2 = roc_points[i + 1]
    tpr_array.append([point1[0], point2[0]])
    fpr_array.append([point1[1], point2[1]])

# We use Trapezoidal rule to calculate the area under the curve and approximating the intergral
auc = sum(np.trapz(tpr_array, fpr_array)) + 1
print('Area under curve={}'.format(auc))
plot_roc(tpr_array, fpr_array, auc)

# pred = clf.predict(X_test)
y_score = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)
plot_roc(fpr, tpr, auc)

fpr, tpr, thresholds = metrics.roc_curve(y_test, clf.predict(X_test), pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)
plot_roc(fpr, tpr, auc)