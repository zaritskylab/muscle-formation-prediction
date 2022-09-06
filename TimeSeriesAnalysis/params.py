import numpy as np

# intensity
window_size = 16
tracks_len = 30
diff_window = [140, 170]
# con_window = [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]
con_window = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]
local_density = False

wt_cols = [wt for wt in range(0, 260, tracks_len)]


def get_tine_windows(start, track_len):
    diff_window = [start, start + track_len]
    con_window = [[x, x + track_len] for x in range(0, 260, track_len)]
    return diff_window, con_window


param_set_trivial = {"window_size": 16,
                     "tracks_len": 30,
                     "start": 140,
                     "diff_window": [140, 170],
                     "con_window": [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
                     "local_density": False,
                     }

param_set_win_size = {"window_size": np.arange(11, 20),
                      "tracks_len": 30,
                      "start": 140,
                      "diff_window": [140, 170],
                      "con_window": [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
                      "local_density": False,
                      }

param_set_track_len = {"window_size": 16,
                       "tracks_len": np.arange(10, 50, 5),
                       "start": 140,
                       "diff_window": [140, 170],
                       "con_window": [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]],
                       "local_density": False,
                       }

PARAMS_DICT = {"window_size": param_set_win_size,
               "tracks_len": param_set_track_len,
               }

registration_method = "reg_MeanOpticalFlowRegistration_"
impute_func = "impute_zeroes"
impute_methodology = "ImputeAllData"
