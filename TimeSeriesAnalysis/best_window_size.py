from tqdm import tqdm

from TimeSeriesAnalysis.mast_intensity import get_intensity_measures_df, open_dirs, get_single_measure_vector_df, \
    plot_intensity_over_time
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    window_size = 60
    n = 10
    l = 200

    dif_window = [140, 170]
    con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]

    dir_name = f"16-01-2022-manual_mastodon_motility-False_intensity-True"
    second_dir = f"{dif_window[0]},{dif_window[1]} frames ERK, {con_windows} frames con"
    open_dirs(dir_name, second_dir)
    dir_name += "/" + second_dir

    for window_size in tqdm([30, 35, 40, 45, 50, 55, 60, 65]):
        int_measures_s5 = get_intensity_measures_df(
            csv_path=r"../data/mastodon/train/Nuclei_5-vertices.csv",
            video_actin_path=r"../data/videos/train/S5_Actin.tif",
            window_size=window_size)
        int_measures_s1 = get_intensity_measures_df(
            csv_path=r"../data/mastodon/train/Nuclei_1-vertices.csv",
            video_actin_path=r"../data/videos/train/S1_Actin.tif",
            window_size=window_size)
        int_measures_s2 = get_intensity_measures_df(
            csv_path=r"../data/mastodon/test/Nuclei_2-vertices.csv",
            video_actin_path=r"../data/videos/test/S2_Actin.tif",
            window_size=window_size)
        int_measures_s3 = get_intensity_measures_df(
            csv_path=r"../data/mastodon/test/Nuclei_3-vertices.csv",
            video_actin_path=r"../data/videos/test/S3_Actin.tif",
            window_size=window_size)

        measure_name = "mean"
        mean_intensity_s2 = get_single_measure_vector_df(intentsity_measures_df=int_measures_s2, measure_name=measure_name)
        mean_intensity_s3 = get_single_measure_vector_df(intentsity_measures_df=int_measures_s3, measure_name=measure_name)
        save_path = dir_name + f"/win_size {window_size}_mean intensity over time S3, S2.png"
        plot_intensity_over_time(measure_name, control_df=mean_intensity_s2, erk_df=mean_intensity_s3, path=save_path)

        mean_intensity_s5 = get_single_measure_vector_df(intentsity_measures_df=int_measures_s5, measure_name=measure_name)
        mean_intensity_s1 = get_single_measure_vector_df(intentsity_measures_df=int_measures_s1, measure_name=measure_name)
        save_path = dir_name + f"/mean intensity over time S5, S1 win_size {window_size}.png"
        plot_intensity_over_time(measure_name, control_df=mean_intensity_s1, erk_df=mean_intensity_s5, path=save_path)

