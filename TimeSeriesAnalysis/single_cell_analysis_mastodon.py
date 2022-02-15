import os
import pickle

import pandas as pd
from TimeSeriesAnalysis.mastodon import load_data, drop_columns, normalize_tracks, get_prob_over_track
from TimeSeriesAnalysis.mastodon_interpretation import get_measueres

if __name__ == '__main__':

    lst_v = [1, 5]  # 1 = Control, 5 = ERK
    video_diff_num = 5  # 8
    video_con_num = 1  # 3
    dif_window = [140, 170]
    motility = True
    visual = False
    con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]

    csv_path = fr"muscle-formation-diff/data/mastodon/" if os.path.exists(
        "muscle-formation-diff/data/mastodon/") else fr"../data/mastodon/Nuclei_{video_diff_num}-vertices.csv"
    dir_name = f"21-12-2021-manual_mastodon_motility-{motility}_intensity-{visual}"
    time_frame = f"{dif_window[0]},{dif_window[1]} frames ERK, {con_windows} frames con"
    complete_path = dir_name + "/" + time_frame

    data = pickle.load(open(
        "21-12-2021-manual_mastodon_motility-True_intensity-False/140,170 frames ERK, [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]] frames con/single_cell_data, motility=True, intensity=False, video #5_2",
        'rb'))

    import matplotlib.pyplot as plt
    import numpy as np

    data["t"] = data.index * 30 * 5 / 60 / 60 * 2
    x = [11 for i in range(0, 10)]
    plt.title("Single treated cell's differentiation confidence over time")
    plt.plot(data["t"], data["w_confidence"])
    plt.plot(x, np.arange(0.0, 1.0, 0.1))
    plt.grid()
    plt.show()

    # # load the model & train set & test set
    clf, X_train, X_test, y_train, y_test = load_data(complete_path)

    df = pd.read_csv(csv_path, encoding="cp1252")
    all_data = pd.DataFrame()

    for label, label_df in df.groupby('Spot track ID'):
        if len(label_df) > 200:
            track = drop_columns(label_df, motility=motility, visual=visual)
            track = normalize_tracks(track, motility=motility, visual=visual)
            track_diff_confidence = get_prob_over_track(clf=clf, track=track, window=30, features_df=X_test,
                                                        moving_window=True)

            all_data = pd.concat(
                [all_data, pd.DataFrame({"w_confidence": track_diff_confidence, "label": label})], axis=0)

    pickle.dump(all_data, open(
        complete_path + "/" + f"moving window confidence, motility={motility}, intensity={visual}, video #{video_diff_num}",
        'wb'))

    # from skimage import io
    # import matplotlib.pyplot as plt
    # bf_video = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\06102021\ERK\S5_Actin.tif"
    # im = io.imread(bf_video)
    #
    #
    # i=200

    # x = int(label_df.iloc[i]["Spot position X (µm)"])
    # y = int(label_df.iloc[i]["Spot position Y (µm)"])
    # current_time = label_df.iloc[i]["Spot frame"]
    # current_label = label_df.iloc[i]["Spot track ID"]
    #
    # fig = plt.figure()
    # plt.imshow(im[i])
    # plt.scatter(x, y, color="none", edgecolor="red")
    # plt.show()
