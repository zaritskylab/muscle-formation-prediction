import pickle

import pandas as pd

from calc_delta_mastodon import calc_prob_delta, plot_avg_diff_prob
from mast_intensity import plot_intensity_over_time, \
    get_intensity_measures_df, load_data, open_dirs, get_intensity_measures_df_df


def get_fusion_time(fusion_times, label):
    lab_df = fusion_times[fusion_times["Spot track ID"] == label]
    cols = list(lab_df.columns)
    cols.remove("manual")
    cols.remove("last_position")
    fusion_time = 0
    for col in cols:
        if lab_df[col].mean() == 1:
            fusion_time = int(col)
    return fusion_time


def get_single_measure_vector_df(intentsity_measures_df, measure_name, fusion_times):
    single_measure_df = pd.DataFrame(columns=[i for i in range(262)])
    for lable, lable_df in intentsity_measures_df.groupby("label"):
        fusion_t = get_fusion_time(fusion_times, int(lable))
        fusion_t = fusion_t if fusion_t != 0 else lable_df["frame"].max()
        frame_label_df = lable_df[["frame", measure_name]]
        frame_label_df.index = frame_label_df["frame"].astype(int)
        frame_label_df = frame_label_df[~frame_label_df.index.duplicated()]
        frame_label_df = frame_label_df[frame_label_df["frame"] < fusion_t]
        frame_label_df["mean"] = frame_label_df["mean"] - frame_label_df["mean"].iloc[0]
        single_measure_df = single_measure_df.append(frame_label_df["mean"])

    single_measure_df.index = [i for i in range(len(single_measure_df))]
    return single_measure_df


# int_measures_s3_fusion = get_intensity_measures_df(
#     csv_path=r"../data/mastodon/test/s3_fusion-vertices.csv",
#     video_actin_path=r"../data/videos/test/S3_Actin.tif",
#     window_size=40)
#
# pos_df = int_measures_s3_fusion[int_measures_s3_fusion["label"].isin(
#     [8987, 19929, 506, 8787, 11135, 8966, 22896, 8594, 14202, 14646, 20010, 7641, 17902, 20043])]
# neg_df = int_measures_s3_fusion[
#     int_measures_s3_fusion["label"].isin([12421, 20455, 26219, 27930, 15645, 14625, 17513, 2590, 30007, 15281, 29878])]
#

# single_measure_df_pos = get_single_measure_vector_df(pos_df, "mean", fusion_times)
# single_measure_df_neg = get_single_measure_vector_df(neg_df, "mean", fusion_times)
# plot_intensity_over_time("mean", control_df=single_measure_df_neg, erk_df=single_measure_df_pos, path="pos_neg_S3.png")


# mean_intensity_pos_df = get_single_measure_vector_df(intentsity_measures_df=pos_df, measure_name="mean")
# mean_intensity_neg_df = get_single_measure_vector_df(intentsity_measures_df=neg_df, measure_name="mean")

def trim_fusion(df):
    fused_df = pd.DataFrame()
    not_fused_df = pd.DataFrame()
    for label, label_df in df.groupby("Spot track ID"):
        fusion_t = get_fusion_time(fusion_times, int(label))
        if fusion_t != 0:  # fused
            label_df = label_df[label_df["Spot frame"] <= fusion_t]
            fused_df = fused_df.append(label_df)
        else:  # not fused at the end of the video
            not_fused_df = not_fused_df.append(label_df)
    return fused_df, not_fused_df


def get_tracks_list(int_df):
    int_df['target'] = 1
    tracks = list()
    for label, labeld_df in int_df.groupby('Spot track ID'):
        tracks.append(labeld_df)
    return tracks


def run(dir_name, fused_df, not_fused_df):
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)
    fused_int_df = get_intensity_measures_df_df(fused_df, r"muscle-formation-diff/data/videos/test/S3_Actin.tif", bounded_window_size)
    not_fused_int_df = get_intensity_measures_df_df(not_fused_df, r"muscle-formation-diff/data/videos/test/S3_Actin.tif", bounded_window_size)

    fused_tracks = get_tracks_list(fused_int_df)
    not_fused_tracks = get_tracks_list(not_fused_int_df)

    prob_fuse = calc_prob_delta(30, fused_tracks, clf, X_test, False, True, wt_cols)
    prob_not_fuse = calc_prob_delta(30, not_fused_tracks, clf, X_test, False, True, wt_cols)

    pickle.dump(prob_fuse, open(dir_name + "/" + f"df_prob_fuse_w={30}, video_num={3}", 'wb'))
    pickle.dump(prob_not_fuse, open(dir_name + "/" + f"df_prob_not_fuse_w={30}, video_num={3}", 'wb'))


if __name__ == '__main__':
    csv_path = fr"muscle-formation-diff/data/mastodon/test/s3_fusion-vertices.csv"
    fusion_times = pd.read_csv(csv_path, encoding="ISO-8859-1")

    fused_df, not_fused_df = trim_fusion(fusion_times)

    # load the model & train set & test set
    wt_cols = [wt for wt in range(0, 260, 30)]

    bounded_window_size = 40
    dir_name = f"15-02-2022-manual_mastodon_motility-False_intensity-True"
    second_dir = f"140,170 frames ERK, [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]] frames con,40 winsize"
    open_dirs(dir_name, second_dir)
    dir_name += "/" + second_dir

    run(dir_name, fused_df, not_fused_df)

    dir_name = f"15-02-2022-manual_mastodon_motility-False_intensity-True"
    second_dir = f"200,230 frames ERK, [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]] frames con,40 winsize"
    open_dirs(dir_name, second_dir)
    dir_name += "/" + second_dir

    run(dir_name, fused_df, not_fused_df)


    # df = pickle.load(open(dir_name + "/" + f"df_prob_w={30}, video_num={3}", 'rb'))

