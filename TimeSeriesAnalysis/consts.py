vid_path_s3_nuc = r"/data/videos/test/s3.tif"
vid_path_s2_nuc = r"/data/videos/test/s2.tif"
vid_path_s5_nuc = r"/data/videos/train/s5.tif"
vid_path_s1_nuc = r"/data/videos/train/s1.tif"
vid_path_s6_nuc = r"/data/videos/train/s6.tif"
vid_path_s7_nuc = r"/data/videos/train/s7.tif"
vid_path_s8_nuc = r"/data/videos/train/s8.tif"
vid_path_s10_nuc = r"/data/videos/train/s10.tif"

vid_path_s5_actin = r"/data/videos/train/S5_Actin.tif"
vid_path_s1_actin = r"/data/videos/train/S1_Actin.tif"
vid_path_s2_actin = r"/data/videos/test/S2_Actin.tif"
vid_path_s3_actin = r"/data/videos/test/S3_Actin.tif"
vid_path_s7_actin = r"/data/videos/test/S7_Actin.tif"
vid_path_s10_actin = r"/data/videos/test/S10_Actin.tif"
vid_path_s6_actin = r"/data/videos/test/S6_Actin.tif"
vid_path_s8_actin = r"/data/videos/test/S8_Actin.tif"

csv_all_s3 = r"/data/mastodon/all_detections_s3-vertices.csv"
csv_all_s2 = r"/data/mastodon/all_detections_s2-vertices.csv"
csv_all_s1 = r"/data/mastodon/all_detections_s1-vertices.csv"
csv_all_s5 = r"/data/mastodon/all_detections_s5-vertices.csv"
csv_all_s7 = r"/data/mastodon/all_detections_s7-vertices.csv"
csv_all_s10 = r"/data/mastodon/all_detections_s10-vertices.csv"
csv_all_s6 = r"/data/mastodon/all_detections_s6-vertices.csv"
csv_all_s8 = r"/data/mastodon/all_detections_s8-vertices.csv"

csv_path_s3 = fr"/data/mastodon/test/Nuclei_3-vertices.csv"
csv_path_s2 = fr"/data/mastodon/test/Nuclei_2-vertices.csv"
csv_path_s5 = fr"/data/mastodon/train/Nuclei_5-vertices.csv"
csv_path_s1 = fr"/data/mastodon/train/Nuclei_1-vertices.csv"
csv_path_s7 = r"/data/mastodon/all_detections_s7-vertices.csv"
csv_path_s10 = r"/data/mastodon/all_detections_s10-vertices.csv"
csv_path_s6 = r"/data/mastodon/all_detections_s6-vertices.csv"
csv_path_s8 = r"/data/mastodon/all_detections_s8-vertices.csv"

cluster_path = "muscle-formation-diff"
local_path = ".."

# intensity
window_size = 15

diff_window = [140, 170]
# con_window = [[10, 40], [40, 70], [70, 100], [100, 130], [130, 160], [160, 190], [190, 220], [220, 250]]
con_window = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]
tracks_len = 30
to_run = "motility"

wt_cols = [wt for wt in range(0, 260, tracks_len)]

s1 = {"name": "S1",
      "target": 0,
      "csv_all_path": csv_all_s1,
      "csv_tagged_path": csv_path_s1,
      "actin_path": vid_path_s1_actin,
      "nuc_path": vid_path_s1_nuc}

s2 = {"name": "S2",
      "target": 0,
      "csv_all_path": csv_all_s2,
      "csv_tagged_path": csv_path_s2,
      "actin_path": vid_path_s2_actin,
      "nuc_path": vid_path_s2_nuc}

s3 = {"name": "S3",
      "target": 1,
      "csv_all_path": csv_all_s3,
      "csv_tagged_path": csv_path_s3,
      "actin_path": vid_path_s3_actin,
      "nuc_path": vid_path_s3_nuc}

s5 = {"name": "S5",
      "target": 1,
      "csv_all_path": csv_all_s5,
      "csv_tagged_path": csv_path_s5,
      "actin_path": vid_path_s5_actin,
      "nuc_path": vid_path_s5_nuc}

s7 = {"name": "S7",
      "target": 1,
      "csv_all_path": csv_all_s7,
      "csv_tagged_path": csv_path_s7,
      "actin_path": vid_path_s7_actin,
      "nuc_path": vid_path_s7_nuc}

s10 = {"name": "S10",
       "target": 1,
       "csv_all_path": csv_all_s10,
       "csv_tagged_path": csv_path_s10,
       "actin_path": vid_path_s10_actin,
      "nuc_path": vid_path_s10_nuc}

s6 = {"name": "S6",
       "target": 1,
       "csv_all_path": csv_all_s6,
       "csv_tagged_path": csv_path_s6,
       "actin_path": vid_path_s6_actin,
      "nuc_path": vid_path_s6_nuc}

s8 = {"name": "S8",
       "target": 1,
       "csv_all_path": csv_all_s8,
       "csv_tagged_path": csv_path_s8,
       "actin_path": vid_path_s8_actin,
      "nuc_path": vid_path_s8_nuc}

