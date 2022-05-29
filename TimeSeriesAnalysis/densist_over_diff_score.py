import pickle
import numpy as np
import matplotlib.pyplot as plt
from TimeSeriesAnalysis.consts import *
import pandas as pd
import matplotlib.cm as cm

from TimeSeriesAnalysis.intensity_erk_compare import get_tracks_list


def load_clean_rows(csv_path):
    df = pd.read_csv(csv_path, encoding="cp1252")
    df = df.drop(labels=range(0, 2), axis=0)
    return df


def get_density(df, experiment):
    densities = pd.DataFrame()
    for t, t_df in df.groupby("Spot frame"):
        densities = densities.append({"Spot frame": t, "density": len(t_df)}, ignore_index=True)
    densities["experiment"] = experiment
    return densities


def get_local_density(df, x, y, t, neighboring_distance):
    neighbors = df[(np.sqrt(
        (df["Spot position X (µm)"] - x) ** 2 + (df["Spot position Y (µm)"] - y) ** 2) <= neighboring_distance) &
                   (df['Spot frame'] == t) &
                   (0 < np.sqrt((df["Spot position X (µm)"] - x) ** 2 + (df["Spot position Y (µm)"] - y) ** 2))]
    return len(neighbors)


def get_tracks(path):
    df_s = load_clean_rows(path)
    df_s.rename(columns={"Spot position": "Spot position X (µm)", "Spot position.1": "Spot position Y (µm)"}, inplace=True)
    df_s["Spot frame"] = df_s["Spot frame"].astype(int)
    df_s["Spot position X (µm)"] = df_s["Spot position X (µm)"].astype(float)
    df_s["Spot position Y (µm)"] = df_s["Spot position Y (µm)"].astype(float)
    tracks_s = get_tracks_list(df_s[df_s["manual"] == 1], target=1)

    return df_s, tracks_s


def get_avg_prob(path):
    probs_s = pickle.load(open(path, 'rb'))
    avg_vals_diff = ([probs_s[col].mean() for col in probs_s.columns])
    std_vals_diff = np.asarray([probs_s[col].std() for col in probs_s.columns])
    p_std = avg_vals_diff + std_vals_diff
    m_std = avg_vals_diff - std_vals_diff
    return p_std, m_std, avg_vals_diff


def get_local_densities_df(df_s, tracks_s, neighboring_distance):
    local_densities = pd.DataFrame(columns=[i for i in range(df_s["Spot frame"].max() + 2)])
    for track in tracks_s:
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        track_local_density = {
            t: get_local_density(df=df_s,
                                 x=track[track["Spot frame"] == t]["Spot position X (µm)"].values[0],
                                 y=track[track["Spot frame"] == t]["Spot position Y (µm)"].values[0],
                                 t=t,
                                 neighboring_distance=neighboring_distance)
            for t in spot_frames}
        local_densities = local_densities.append(track_local_density, ignore_index=True)
    return local_densities


def plot_diff_score_over_density(path, mean_density, color, label):
    p_std, m_std, avg_vals_diff = get_avg_prob(path)
    scatter = plt.scatter(x=list(mean_density.values())[:len(avg_vals_diff)], y=avg_vals_diff,
                          c=np.array(list(mean_density.keys())[:len(avg_vals_diff)]) * 5 / 60, marker=".",
                          cmap=cm.get_cmap(color, 512), label=label)
    plt.colorbar(scatter, orientation='horizontal')
    # plt.scatter(np.array(list(mean_density.values())[:len(avg_vals_diff)]), avg_vals_diff, color=color)


if __name__ == '__main__':
    path = local_path

    neighboring_distance = 50

    df_s3, tracks_s3 = get_tracks(path + csv_all_s3)
    df_s2, tracks_s2 = get_tracks(path + csv_all_s2)
    df_s5, tracks_s5 = get_tracks(path + csv_all_s5)
    df_s1, tracks_s1 = get_tracks(path + csv_all_s1)


    df_s2.rename(columns={"Spot position": "Spot position X (µm)", "Spot position.1": "Spot position Y (µm)"}, inplace=True)

    local_densities_s3 = get_local_densities_df(df_s3, tracks_s3, neighboring_distance)
    local_densities_s2 = get_local_densities_df(df_s2, tracks_s2, neighboring_distance)
    local_densities_s5 = get_local_densities_df(df_s5, tracks_s5, neighboring_distance)
    local_densities_s1 = get_local_densities_df(df_s1, tracks_s1, neighboring_distance)

    mean_density_s3 = {col: np.nanmean(local_densities_s3[col]) for col in local_densities_s3.columns}
    mean_density_s2 = {col: np.nanmean(local_densities_s2[col]) for col in local_densities_s2.columns}
    mean_density_s5 = {col: np.nanmean(local_densities_s5[col]) for col in local_densities_s5.columns}
    mean_density_s1 = {col: np.nanmean(local_densities_s1[col]) for col in local_densities_s1.columns}

    dir_path = f"20-03-2022-manual_mastodon_{to_run} shifted tracks"
    second_dir = f"{diff_window} frames ERK, {con_window} frames con track len {tracks_len}"
    dir_path += "/" + second_dir
    plot_diff_score_over_density(dir_path + "/" + "df_prob_w=30, video_num=3", mean_density_s3, "Orange", "s3")
    plot_diff_score_over_density(dir_path + "/" + "df_prob_w=30, video_num=2", mean_density_s2, "Blue", "s2")

    dir_path = f"20-03-2022-manual_mastodon_{to_run} shifted tracks s2,s3 train"
    second_dir = f"{diff_window} frames ERK, {con_window} frames con track len {tracks_len}"
    dir_path += "/" + second_dir
    plot_diff_score_over_density(dir_path + "/" + "df_prob_w=30, video_num=5", mean_density_s5, "wheat", "s5")
    plot_diff_score_over_density(dir_path + "/" + "df_prob_w=30, video_num=1", mean_density_s1, "steelblue", "s1")

    plt.xlabel(f"avg lcoal density -{neighboring_distance}")
    plt.ylabel("avg differentiation score")
    plt.title("lcoal density over time")
    plt.legend()
    plt.show()
