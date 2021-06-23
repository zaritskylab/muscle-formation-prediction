import itertools
import os
import pickle
import numpy as np
import joblib

from ts_fresh import get_x_y, short_extract_features, extract_distinct_features, train, plot_roc, build_pca, plot_pca, \
    feature_importance

if __name__ == '__main__':
    print("Let's go!")

    motility = False
    intensity = True
    min_length = 0
    max_length = 950
    min_time_diff = 0

    exp_1 = [1, 2, 3, 4]
    exp_2 = [5, 6, 7, 8]
    exp_3 = [9, 10, 11, 12]
    # exp_1=[1]
    # exp_2= [3]
    # exp_3= [1,3]

    train_video_lists = [list(itertools.chain(exp_1, exp_2)),
                         list(itertools.chain(exp_1, exp_3)), list(itertools.chain(exp_2, exp_3))]
    test_video_lists = [exp_3, exp_2, exp_1]

    auc_scores = []

    for (exp_train_lst, exp_test_lst) in zip(train_video_lists, test_video_lists):

        dir_name = f"train_set {exp_train_lst}_ test_set {exp_test_lst}_ motility-{motility}_intensity-{intensity}"
        print(dir_name)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        X_train, y_train = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff,
                                   lst_videos=exp_train_lst, motility=motility, intensity=intensity)
        # extract features using ts-fresh
        X_train = short_extract_features(X_train, y_train)

        X_test, y_test = get_x_y(min_length=min_length, max_length=max_length, min_time_diff=min_time_diff,
                                 lst_videos=exp_test_lst, motility=motility, intensity=intensity)
        X_test = extract_distinct_features(df=X_test, feature_list=X_train.columns)

        clf, report, auc_score = train(X_train, X_test, y_train, y_test)
        plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)
        principal_df, pca = build_pca(3, X_test)
        plot_pca(principal_df, pca, dir_name)
        feature_importance(clf, X_train.columns, dir_name)

        auc_scores.append(auc_score)

        pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
        pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
        joblib.dump(clf, dir_name + "/" + "clf.joblib")

        txt_file = open(dir_name + '/info.txt', 'a')
        txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
        txt_file.close()

    print(f"AVG AUC score: {np.mean(auc_scores)}")
