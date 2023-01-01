# intensity
import math

window_size = 16 # window crop on actin intensity measurements
window_size_arr = [10, 13, 16, 19, 21, 24, 27, 30]  # window crop on actin intensity measurements
tracks_len = 30  # number of frames to train the model on
tracks_len_arr = [6, 12, 18, 24, 30, 36, 42]
interval_track_arr = [5, 5, 5, 8, 12, 15, 10]
diff_window = [130, 160]  # frames ro train the model on [140, 170]
con_window = [[0, 30], [40, 70], [90, 120], [130, 160], [180, 210], [220, 250]]  # [140, 170] frames ro train the model
local_density = False

# wt_cols = [wt for wt in range(0, 260, tracks_len)]

registration_method = "MeanOpticalFlowReg_"
impute_func = "impute_zeroes"
impute_methodology = "ImputeAllData"


def create_temporal_segment(segment_arr):
    temporal_seg_arr = segment_arr
    diff_window_arr = [[160 - track_len, 160] for track_len in temporal_seg_arr]
    con_window_arr = []
    for tup_len_interval in zip(temporal_seg_arr, interval_track_arr):
        con_window_specify = []
        track_len = tup_len_interval[0]
        interval = tup_len_interval[1]

        start_win = 0
        while start_win + track_len <= 250:
            con_window_specify.append([start_win, start_win + track_len])
            start_win += (track_len + interval)

        con_window_arr.append(con_window_specify)

    return temporal_seg_arr, diff_window_arr, con_window_arr

#
# track_len, diff, con = create_temporal_segment([6, 12, 18, 24, 30, 36, 42])


feature_calc_types = {
    "window_size_arr": window_size_arr,
    "temporal_segment_arr": [create_temporal_segment(tracks_len_arr)]
}