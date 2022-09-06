
vid_path_s3_nuc = r"data/videos/test/S3_nuclei.tif"
vid_path_s2_nuc = r"data/videos/test/S2_nuclei.tif"
vid_path_s5_nuc = r"data/videos/train/S5_nuclei.tif"
vid_path_s1_nuc = r"data/videos/train/S1_nuclei.tif"
vid_path_s6_nuc = r"data/videos/test/S6_nuclei.tif"
vid_path_s7_nuc = r"data/videos/train/s7.tif"
vid_path_s8_nuc = r"data/videos/test/S8_nuclei.tif"
vid_path_s1_ck666_nuc = r"data/videos/test/211213_s1_control_Nuclei.tif"
vid_path_s4_ck666_nuc = r"data/videos/test/211213_s4_ERKi_Nuclei.tif"
vid_path_s5_ck666_nuc = r"data/videos/test/211213_s5_ERKiCK666150_Nuclei.tif"
vid_path_s10_nuc = r"data/videos/train/s10.tif"

vid_path_s5_actin = r"data/videos/train/S5_Actin.tif"
vid_path_s1_actin = r"data/videos/train/S1_Actin.tif"
vid_path_s2_actin = r"data/videos/test/S2_Actin.tif"
vid_path_s3_actin = r"data/videos/test/S3_Actin.tif"
vid_path_s7_actin = r"data/videos/test/S7_Actin.tif"
vid_path_s10_actin = r"data/videos/test/S10_Actin.tif"
vid_path_s6_actin = r"data/videos/test/S6_Actin.tif"
vid_path_s8_actin = r"data/videos/test/S8_Actin.tif"
vid_path_s1_ck666_actin = "data/videos/test/211213_s1_control_Actin.tif"
vid_path_s4_ck666_actin = "data/videos/test/211213_s4_ERKi_Actin.tif"
vid_path_s5_ck666_actin = "data/videos/test/211213_s5_ERKiCK666150_Actin.tif"

data_csv_path = r"data/mastodon/%s%s all detections.csv"

cluster_path = "muscle-formation-diff"
local_path = ".."

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

s1_211212 = {"name": "S1 211212",
             "target": 0,
             "actin_path": "not exist",
             "nuc_path": "not exist"}

s4_211212 = {"name": "S4 211212",
             "target": 1,
             "actin_path": "not exist",
             "nuc_path": "not exist"}

s5_211212 = {"name": "S5 211212",  # erk + p38
             "target": 1,
             "actin_path": "not exist",
             "nuc_path": "not exist"}

s7_211212 = {"name": "S7 211212",  # p38
             "target": 0,
             "actin_path": "not exist",
             "nuc_path": "not exist"}

s1_ck666 = {"name": "S1 ck666",  # control
            "target": 0,
            "actin_path": vid_path_s1_ck666_actin,
            "nuc_path": vid_path_s1_ck666_nuc}

s4_ck666 = {"name": "S4 ck666",  # ERKi positive control
            "target": 1,
            "actin_path": vid_path_s4_ck666_actin,
            "nuc_path": vid_path_s4_ck666_nuc}

s5_ck666 = {"name": "S5 ck666",  # ERKi ck666
            "target": 1,
            "actin_path": vid_path_s5_ck666_actin,
            "nuc_path": vid_path_s5_ck666_nuc}

s_runs = {"1": s1,
          "2": s2,
          "3": s3,
          "5": s5,
          "6": s6,
          "8": s8,
          "s1_211212": s1_211212,
          "s4_211212": s4_211212,
          "s5_211212": s5_211212,
          "s7_211212": s7_211212,
          "s1_ck666": s1_ck666,
          "s5_ck666": s5_ck666,
          "s4_ck666": s4_ck666,

          "11": s1_ck666,
          "55": s5_ck666,
          "44": s4_ck666,

          }
