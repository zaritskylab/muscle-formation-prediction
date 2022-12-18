# intensity
window_size = 16 # window crop on actin intensity measurements
window_size_arr = [10, 13, 16, 19, 21, 24, 27, 30]  # window crop on actin intensity measurements
tracks_len = 30  # number of frames to train the model on
diff_window = [140, 170]  # frames ro train the model on
con_window = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]] # frames ro train the model on
local_density = False
wt_cols = [wt for wt in range(0, 260, tracks_len)]

registration_method = "MeanOpticalFlowReg_"
impute_func = "impute_zeroes"
impute_methodology = "ImputeAllData"

feature_calc_types = {
    "window_size_arr": window_size_arr
}