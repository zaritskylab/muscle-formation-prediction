storage_path = r"/storage/users/assafzar/Muscle_Differentiation_AvinoamLab/"
data_csv_path = storage_path + r"data/mastodon/%s%s all detections.csv"

intensity_model_path = storage_path + r"15-12-2022-actin_intensity local dens-False, s%s, s%s train [130, 160] diff window win size 16/track len 30, impute_func-ImputeAllData_impute_zeroes reg MeanOpticalFlowReg_/"
motility_model_path = storage_path + r"15-12-2022-motility local dens-False, s%s, s%s train [130, 160] diff window/track len 30, impute_func-ImputeAllData_impute_zeroes reg MeanOpticalFlowReg_/"
cluster_path = "muscle-formation-diff"
local_path = ".."

PIXEL_SIZE = 0.462
FEATURES_DIR_PATH = f"data/mastodon/features/"

s1 = {"name": "S1",
      "target": 0,
      "actin_path": storage_path + r"data/videos/S1_Actin.tif",
      "nuc_path": storage_path + r"data/videos/S1_nuclei.tif",
      }

s2 = {"name": "S2",
      "target": 0,
      "actin_path": storage_path + r"data/videos/S2_Actin.tif",
      "nuc_path": storage_path + r"data/videos/S2_nuclei.tif",
      }

s3 = {"name": "S3",
      "target": 1,
      "actin_path": storage_path + r"data/videos/S3_Actin.tif",
      "nuc_path": storage_path + r"data/videos/S3_nuclei.tif",
      }

s5 = {"name": "S5",
      "target": 1,
      "actin_path": storage_path + r"data/videos/S5_Actin.tif",
      "nuc_path": storage_path + r"data/videos/S5_nuclei.tif",
      }

s6 = {"name": "S6",
      "target": 1,
      "actin_path": storage_path + r"data/videos/S6_Actin.tif",
      "nuc_path": storage_path + r"data/videos/S6_nuclei.tif"}

s7 = {"name": "S7",
      "target": 1,
      "actin_path": storage_path + r"data/videos/S7_Actin.tif",
      "nuc_path": storage_path + r"data/videos/s7.tif"}

s10 = {"name": "S10",
       "target": 1,
       "actin_path": storage_path + r"data/videos/S10_Actin.tif",
       "nuc_path": storage_path + r"data/videos/s10.tif"}

s8 = {"name": "S8",
      "target": 1,
      "actin_path": storage_path + r"data/videos/S8_Actin.tif",
      "nuc_path": storage_path + r"data/videos/S8_nuclei.tif"}

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
            "actin_path": storage_path + "data/videos/211213_s1_control_Actin.tif",
            "nuc_path": storage_path + r"data/videos/211213_s1_control_Nuclei.tif"}

s4_ck666 = {"name": "S4 ck666",  # ERKi positive control
            "target": 1,
            "actin_path": storage_path + "data/videos/211213_s4_ERKi_Actin.tif",
            "nuc_path": storage_path + r"data/videos/211213_s4_ERKi_Nuclei.tif"}

s5_ck666 = {"name": "S5 ck666",  # ERKi ck666
            "target": 1,
            "actin_path": storage_path + "data/videos/211213_s5_ERKiCK666150_Actin.tif",
            "nuc_path": storage_path + r"data/videos/211213_s5_ERKiCK666150_Nuclei.tif"}

vid_info_dict = {"1": s1,
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
