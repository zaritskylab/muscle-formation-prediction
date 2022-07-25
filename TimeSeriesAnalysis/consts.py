vid_path_s3_nuc = r"data/videos/test/S3_nuclei.tif"
vid_path_s2_nuc = r"data/videos/test/S2_nuclei.tif"
vid_path_s5_nuc = r"data/videos/train/S5_nuclei.tif"
vid_path_s1_nuc = r"data/videos/train/S1_nuclei.tif"
vid_path_s6_nuc = r"data/videos/train/s6.tif"
vid_path_s7_nuc = r"data/videos/train/s7.tif"
vid_path_s8_nuc = r"data/videos/train/s8.tif"
vid_path_s10_nuc = r"data/videos/train/s10.tif"

vid_path_s5_actin = r"data/videos/train/S5_Actin.tif"
vid_path_s1_actin = r"data/videos/train/S1_Actin.tif"
vid_path_s2_actin = r"data/videos/test/S2_Actin.tif"
vid_path_s3_actin = r"data/videos/test/S3_Actin.tif"
vid_path_s7_actin = r"data/videos/test/S7_Actin.tif"
vid_path_s10_actin = r"data/videos/test/S10_Actin.tif"
vid_path_s6_actin = r"data/videos/test/S6_Actin.tif"
vid_path_s8_actin = r"data/videos/test/S8_Actin.tif"

data_csv_path = r"data/mastodon/%s%s all detections.csv"

cluster_path = "muscle-formation-diff"
local_path = ".."

# intensity
window_size = 16
tracks_len = 30

diff_window = [140, 170]
# con_window = [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]
con_window = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

to_run = "motility"
local_density = False

wt_cols = [wt for wt in range(0, 260, tracks_len)]

s1 = {"name": "S1",
      "target": 0,
      "actin_path": vid_path_s1_actin,
      "nuc_path": vid_path_s1_nuc,
      }

s2 = {"name": "S2",
      "target": 0,
      "actin_path": vid_path_s2_actin,
      "nuc_path": vid_path_s2_nuc,
      }

s3 = {"name": "S3",
      "target": 1,
      "actin_path": vid_path_s3_actin,
      "nuc_path": vid_path_s3_nuc,
      }

s5 = {"name": "S5",
      "target": 1,
      "actin_path": vid_path_s5_actin,
      "nuc_path": vid_path_s5_nuc,
      }

s6 = {"name": "S6",
      "target": 1,
      "actin_path": vid_path_s6_actin,
      "nuc_path": vid_path_s6_nuc}

s7 = {"name": "S7",
      "target": 1,
      "actin_path": vid_path_s7_actin,
      "nuc_path": vid_path_s7_nuc}

s10 = {"name": "S10",
       "target": 1,
       "actin_path": vid_path_s10_actin,
       "nuc_path": vid_path_s10_nuc}

s8 = {"name": "S8",
      "target": 1,
      "actin_path": vid_path_s8_actin,
      "nuc_path": vid_path_s8_nuc}

s_runs = {"1": s1,
          "2": s2,
          "3": s3,
          "5": s5,
          "6": s6,
          "8": s8,
          }
