# intensity
window_size = 16
tracks_len = 30
diff_window = [140, 170]
con_window = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]
local_density = False

wt_cols = [wt for wt in range(0, 260, tracks_len)]

registration_method = "MeanOpticalFlowReg_"
impute_func = "impute_zeroes"
impute_methodology = "ImputeAllData"
