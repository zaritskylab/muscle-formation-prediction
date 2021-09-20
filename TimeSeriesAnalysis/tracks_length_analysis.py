import pickle
import joblib
from DataPreprocessing.load_tracks_xml import load_tracks_xml
from TimeSeriesAnalysis.ts_fresh import get_path, get_x_y, short_extract_features, train, build_pca, plot_pca, \
    feature_importance, \
    extract_distinct_features, plot_roc, plot_sampled_cells
import os

if __name__ == '__main__':
    print("Let's go! In this script we examine how different cell tracks length affect the model's performances. \n"
          "We also examine the labeling of the ERK group, by determining the min_time_diff parameter")
    # params
    motility = [True, True]
    intensity = [False, False]
    lst_videos_train = [2, 4, 5, 6, 7, 8]
    lst_videos_test = [1, 3]
    min_length = [0, 20, 100, 200, 200, 200, 300, 400, 400, 300, 400, 300]
    max_length = [950, 950, 950, 250, 300, 300, 350, 450, 400, 650, 800, 920]
    min_time_diff = [0, 500, 300, 550, 550, 400, 300, 300, 300, 300, 300, 300]

    # load ERK's tracks and dataframe
    xml_path_diff = get_path(r"../data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s11_all_pixelratio1.xml")
    bf_video_diff = get_path(
        r"../data/videos/BrightField_pixel_ratio_1/Experiment1_w2Brightfield_s11_all_pixelratio1.tif")
    tracks_diff, df = load_tracks_xml(xml_path_diff)

    # load control's tracks and dataframe
    xml_path_con = get_path(r"../data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s9_all_pixelratio1.xml")
    bf_video_con = get_path(
        r"../data/videos/BrightField_pixel_ratio_1/Experiment1_w2Brightfield_s9_all_pixelratio1.tif")
    tracks_con, df = load_tracks_xml(xml_path_con)

    # iterate through all combinations of:
    # 1. motility/intensity
    # 2. track's minimal length (min_length)
    # 3. track's maximal length (max_length)
    # 4. minimal time point to define ERK's track as "differentiated" (min_time_diff)
    for mot_int in zip(motility, intensity):
        for comb in zip(min_length, max_length, min_time_diff):

            # open a new directory to save the outputs in
            dir_name = f"min_length {comb[0]}_ max_length {comb[1]}_ min_time_diff {comb[2]}_ motility-{mot_int[0]}_intensity-{mot_int[1]}"
            print(dir_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            # generate train data & extract features using tsfresh
            X_train, y_train = get_x_y(min_length=comb[0], max_length=comb[1], min_time_diff=comb[2],
                                       lst_videos=lst_videos_train, motility=mot_int[0], intensity=mot_int[1])
            X_train = short_extract_features(X_train, y_train)

            # generate test data & extract features
            X_test, y_test = get_x_y(min_length=comb[0], max_length=comb[1], min_time_diff=comb[2],
                                     lst_videos=lst_videos_test,
                                     motility=mot_int[0], intensity=mot_int[1])
            X_test = extract_distinct_features(X_test, X_train.columns)  # get only the train set's features

            # train the classifier
            clf, report, auc_score = train(X_train, X_test, y_train, y_test)

            # perform PCA analysis
            principal_df, pca = build_pca(3, X_test)
            plot_pca(principal_df, pca, dir_name)

            # plot ROC curve
            plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)

            # calculate feature importance
            feature_importance(clf, X_train.columns, dir_name)

            # save the model & train set & test set
            pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
            pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
            joblib.dump(clf, dir_name + "/" + "clf.joblib")

            # save classification report & AUC score
            txt_file = open(dir_name + '/info.txt', 'a')
            txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
            txt_file.close()

            # sample cells with relatively long tracks, and plot their probability of being differentiated over time
            # ERK
            plot_sampled_cells(window=40, track_length=600, clf=clf, features_df=X_train, dir_name=dir_name,
                               bf_video=bf_video_diff, tracks=tracks_diff, con_diff="diff", n_cells=10)
            # control
            plot_sampled_cells(window=40, track_length=200, clf=clf, features_df=X_train, dir_name=dir_name,
                               bf_video=bf_video_con, tracks=tracks_con, con_diff="con", n_cells=10)
