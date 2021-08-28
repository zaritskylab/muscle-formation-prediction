import itertools
import os
import pickle
import numpy as np
import joblib

from TimeSeriesAnalysis.ts_fresh import get_x_y, short_extract_features, extract_distinct_features, train
from TimeSeriesAnalysis.ts_interpretation import plot_roc, plot_pca, build_pca, feature_importance

if __name__ == '__main__':
    print("Let's go! In this script, we will consider only the beginning/end of the experiments\n"
          "(~Fisrt 3 hours or last 3 hours)")

    # params
    motility = False
    intensity = True
    min_length = 0
    max_length = 950
    min_time_diff = 0
    auc_scores = []

    # split videos into their experiments
    exp_1 = [1, 2, 3, 4]
    exp_2 = [5, 6, 7, 8]
    exp_3 = [9, 10, 11, 12]


    # create combinations for leave one out training
    train_video_lists = [list(itertools.chain(exp_1, exp_2)),
                         list(itertools.chain(exp_1, exp_3)), list(itertools.chain(exp_2, exp_3))]
    test_video_lists = [exp_3, exp_2, exp_1]

    # iterate through all experiments, performing leave one out
    for (exp_train_lst, exp_test_lst) in zip(train_video_lists, test_video_lists):
        main_dir_name = "outputs/end of the experiment"
        if not os.path.exists(main_dir_name):
            os.mkdir(main_dir_name)

        # open a new directory to save the outputs in
        dir_name = main_dir_name + "/" + f"train_set {exp_train_lst}_ test_set {exp_test_lst}_ motility-{motility}_intensity-{intensity}"
        print(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # generate train data & extract features using tsfresh
        X_train, y_train = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff,
                                   lst_videos=exp_train_lst, motility=motility, intensity=intensity, crop_start=False,
                                   crop_end=True)
        # extract features using ts-fresh
        X_train = short_extract_features(X_train, y_train)

        # generate test data & extract features
        X_test, y_test = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff,
                                 lst_videos=exp_test_lst, motility=motility, intensity=intensity, crop_start=False,
                                 crop_end=True)
        X_test = extract_distinct_features(df=X_test, feature_list=X_train.columns)  # get only the train set's features

        # train the classifier
        clf, report, auc_score = train(X_train, X_test, y_train, y_test)
        auc_scores.append(auc_score)

        # plot ROC curve
        plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)

        # perform PCA analysis
        principal_df, pca = build_pca(3, X_test)
        plot_pca(principal_df, pca, dir_name)

        # calculate feature importance
        feature_importance(clf, X_train.columns, dir_name)

        # save the model & train set & test set
        pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
        pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
        pickle.dump(y_test, open(dir_name + "/" + "y_test", 'wb'))
        pickle.dump(y_train, open(dir_name + "/" + "y_train", 'wb'))
        joblib.dump(clf, dir_name + "/" + "clf.joblib")

        # save classification report & AUC score
        txt_file = open(dir_name + '/info.txt', 'a')
        txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
        txt_file.close()
    # print average AUC score
    print(f"AVG AUC score: {np.mean(auc_scores)}")
