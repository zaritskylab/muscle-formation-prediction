import itertools
import pickle

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve

if __name__ == '__main__':
    print(
        "plot mean roc curve")

    # params
    motility = False
    intensity = True
    min_length = 0
    max_length = 950
    min_time_diff = 0
    auc_scores = []
    window = 40

    # split videos into their experiments
    exp_1 = [1, 2, 3, 4]
    exp_2 = [5, 6, 7, 8]
    exp_3 = [9, 10, 11, 12]

    # create combinations for leave one out training
    train_video_lists = [list(itertools.chain(exp_1, exp_2)),
                         list(itertools.chain(exp_1, exp_3)), list(itertools.chain(exp_2, exp_3))]
    test_video_lists = [exp_3, exp_2, exp_1]
    i = 0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    # iterate through all experiments, performing leave one out
    for (exp_train_lst, exp_test_lst) in zip(train_video_lists, test_video_lists):

        # open the needed directory
        dir_name = f"start of the experiment/train_set {exp_train_lst}_ test_set {exp_test_lst}_ motility-{motility}_intensity-{intensity}"
        print(dir_name)
        print(motility)
        print(intensity)

        # Load classifier
        classifier = clf = joblib.load(dir_name + "/" + "clf.joblib")
        # Load X_test
        X_test = pickle.load(open(dir_name + "/" + "X_test", 'rb'))
        y_test = pickle.load(open(dir_name + "/" + "y_test", 'rb'))

        classifier.fit(X_test, y_test)
        viz = plot_roc_curve(classifier, X_test, y_test,
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    plt.savefig(dir_name + "/" + "Receiver operating characteristic")
    plt.show()
