import pickle

import joblib
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

from DataPreprocessing.load_tracks_xml import load_tracks_xml
from ts_fresh import get_path, get_x_y, short_extract_features, train, build_pca, plot_pca, feature_importance
import os

from ts_interpretation import plot_cell_probability

if __name__ == '__main__':
    print("Let's go!")

    motility = [True, True]
    intensity = [False, False]
    lst_videos_train = [2, 4, 5, 6, 7, 8]
    lst_videos_test = [1, 3]
    min_length = [0, 20, 100, 200, 200, 200, 300, 400, 400, 300, 400, 300]
    max_length = [950, 950, 950, 250, 300, 300, 350, 450, 400, 650, 800, 920]
    min_time_diff = [0, 500, 300, 550, 550, 400, 300, 300, 300, 300, 300, 300]

    combinations = zip(min_length, max_length, min_time_diff)
    motility_intensity = zip(motility, intensity)

    xml_path_diff = get_path(r"data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s11_all_pixelratio1.xml")
    bf_video_diff = get_path(r"data/videos/BrightField_pixel_ratio_1/Experiment1_w2Brightfield_s11_all_pixelratio1.tif")

    xml_path_con = get_path(r"data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s9_all_pixelratio1.xml")
    bf_video_con = get_path(r"data/videos/BrightField_pixel_ratio_1/Experiment1_w2Brightfield_s9_all_pixelratio1.tif")

    tracks_diff, df = load_tracks_xml(xml_path_diff)
    tracks_con, df = load_tracks_xml(xml_path_con)

    for mot_int in motility_intensity:
        for comb in zip(min_length, max_length, min_time_diff):

            dir_name = f"min_length {comb[0]}_ max_length {comb[1]}_ min_time_diff {comb[2]}_ motility-{mot_int[0]}_intensity-{mot_int[1]}"
            print(dir_name)
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)

            X_train, y_train = get_x_y(min_length=comb[0], max_length=comb[1], min_time_diff=comb[2],
                                       lst_videos=lst_videos_train,
                                       motility=mot_int[0], intensity=mot_int[1])
            X_train = short_extract_features(X_train, y_train)

            X_test, y_test = get_x_y(min_length=comb[0], max_length=comb[1], min_time_diff=comb[2],
                                     lst_videos=lst_videos_test,
                                     motility=mot_int[0], intensity=mot_int[1])
            X_test = extract_features(X_test, column_id="label", column_sort="t")
            impute(X_test)
            X_test = X_test[X_train.columns]

            clf, report, auc_score = train(X_train, X_test, y_train, y_test)
            principal_df, pca = build_pca(3, X_test)
            plot_pca(principal_df, pca, dir_name)
            feature_importance(clf, X_train.columns, dir_name)

            pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
            pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
            joblib.dump(clf, dir_name + "/" + "clf.joblib")

            txt_file = open(dir_name + '/info.txt', 'a')
            txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
            txt_file.close()

            count1 = 0
            count2 = 0
            for cell_ind, curr_track in enumerate(tracks_diff):
                if len(curr_track) < 600: continue
                if count1 > 10: continue
                count1 += 1
                print(len(curr_track))
                plot_cell_probability(cell_ind=cell_ind, bf_video=bf_video_diff, clf=clf, track=curr_track, window=40,
                                      target_val=True, features_df=X_train,
                                      text="", path=dir_name + "/dif_" + str(cell_ind))

            for cell_ind, curr_track in enumerate(tracks_con):
                if len(curr_track) < 200: continue
                if count2 > 10: continue
                count2 += 1
                print(len(curr_track))
                plot_cell_probability(cell_ind=cell_ind, bf_video=bf_video_con, clf=clf, track=curr_track, window=40,
                                      target_val=False, features_df=X_train,
                                      text="", path=dir_name + "/control_" + str(cell_ind))
