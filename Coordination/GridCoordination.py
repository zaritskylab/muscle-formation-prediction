from Scripts.Coordination.CoordinationCalc import CoordinationCalc
from Scripts.Coordination.CoordinationGraphBuilder import CoordGraphBuilder
from Scripts.DataPreprocessing.load_tracks_xml import load_tracks_xml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Get the dataframe:
xml_path = r"../../data/tracks_xml/Experiment1_w1Widefield550_s3_all.xml"
tracks01, df = load_tracks_xml(xml_path)

# Divide into grids
n_grids_x = 3
n_grids_y = 3

min_x = np.min(df["x"])
min_y = np.min(df["y"])
max_x = np.max(df["x"])
max_y = np.max(df["y"])

share_x = max_x / n_grids_x
share_y = max_y / n_grids_y

time_slot = 5
coord_arr = np.zeros((int(np.max(df["t"])),n_grids_x, n_grids_y), )
# coord_arr = [[0 for i in range(n_grids_x)] for j in range(n_grids_y)]

for k in range(0, n_grids_x):
    for j in range(0, n_grids_y):
        grid_tracks = []
        grid_i_df = df[(df["x"] <= share_x * (k + 1)) & (share_x * k <= df["x"])
                       & (df["y"] <= share_y * (j + 1)) & (share_y * j <= df["y"])]
        for i in np.arange(0, np.max(grid_i_df["label"]) + 1, 1.0):
            if i in (grid_i_df.label.unique()):
                grid_tracks.append(grid_i_df[grid_i_df["label"] == i])
            else:
                grid_tracks.append(None)
        coord_calculator = CoordinationCalc(SMOOTHING_VAR=5, NEIGHBORING_DISTANCE=0.3)
        coord_calculator.build_coordination_df(grid_tracks, grid_i_df)

        g_builder = CoordGraphBuilder()
        coordination, time_c = g_builder.read_coordinationDF(coord_calculator.coherence)

        mean_coords = []
        for i in range(0, int(np.max(df["t"])) - 1):
            num_of_cells = grid_i_df[(i <= grid_i_df["t"]) & (grid_i_df["t"] < i + time_slot)]["label"].nunique()
            mean_coord_time_slot = np.mean(coordination[i:i + time_slot] / num_of_cells)
            coord_arr[i][k][j] = mean_coord_time_slot


np.save("control_grid.npy")

for i in range(len(coord_arr)):
    fig, ax = plt.subplots()
    # im = ax.imshow(coord_arr[i]-np.mean(coord_arr[i]))
    plt.pcolor(coord_arr[i] / np.sum(coord_arr[i]), cmap=plt.cm.seismic, vmin=-0, vmax=0.3)
    plt.colorbar()
    plt.title("Coordination grid- diff, time = {}".format(i))
    plt.savefig("diff_grid/{}".format(i))
