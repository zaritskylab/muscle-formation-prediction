diff_window = [130, 160]  # frames ro train the model on [140, 170]
diff_window_arr = [[125, 155], [127, 157], [130, 160], [133, 163], [136, 166], [140, 170]]
con_window = [[0, 30], [40, 70], [90, 120], [130, 160], [180, 210], [220, 250]]

window_size_arr = [10, 13, 16, 19, 21, 24, 27, 30]  # window crop on actin intensity measurements
tracks_len_arr = [6, 12, 18, 24, 30, 36, 42]  # arr of temporal segment size
interval_track_arr = [5, 5, 5, 8, 12, 15, 10]  # for each temporal segment size in the location i in the tracks len arr


# have interval size in the same location i in interval track arr
# the interval is to define the size between the con windows of each temporal segment size

# this function create 2 new arrays -> diff_window_arr and con_window arr accordind to the size
# of each temporal segment in the track len arr
def create_temporal_segment(segment_arr):
    # define the temporal_seg_arr
    temporal_seg_arr = segment_arr

    # create the diff_window arr
    diff_window_arr = [[160 - track_len, 160] for track_len in temporal_seg_arr]

    # create the con_window arr
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


feature_calc_types = {
    "window_size_arr": window_size_arr,
    "temporal_segment_arr": [create_temporal_segment(tracks_len_arr)],
    "diff_window_arr": diff_window_arr
}
