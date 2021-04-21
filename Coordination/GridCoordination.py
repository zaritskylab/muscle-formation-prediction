from CoordinationCalc import CoordinationCalc
from CoordinationGraphBuilder import CoordGraphBuilder
from DataPreprocessing.load_tracks_xml import load_tracks_xml
import numpy as np


def get_grid_coordination(xml_path, n_grids_x, n_grids_y):
    # Get the dataframe:
    tracks01, df = load_tracks_xml(xml_path)

    max_x = np.max(df["x"])
    max_y = np.max(df["y"])

    share_x = max_x / n_grids_x
    share_y = max_y / n_grids_y

    time_slot = 5
    coord_arr = np.zeros((int(np.max(df["t_stamp"])), n_grids_x, n_grids_y), )

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
            coord_calculator = CoordinationCalc(SMOOTHING_VAR=5, NEIGHBORING_DISTANCE=30)
            coord_calculator.build_coordination_df_(grid_tracks, grid_i_df)

            g_builder = CoordGraphBuilder()
            coordination, time_c = g_builder.read_coordinationDF(coord_calculator.coherence)

            for i in range(0, int(np.max(df["t_stamp"])) - 1):
                num_of_cells = grid_i_df[(i <= grid_i_df["t_stamp"]) & (grid_i_df["t_stamp"] < i + time_slot)]["label"].nunique()
                slot_coord = coordination[i:i + time_slot]
                slot_coord = slot_coord[~np.isnan(slot_coord)]
                if num_of_cells ==0:
                    mean_coord_time_slot=0
                else:
                    mean_coord_time_slot = np.mean(slot_coord) / num_of_cells
                coord_arr[i][k][j] = mean_coord_time_slot
    return coord_arr



xml_path = r"../data/tracks_xml/0104/Experiment1_w1Widefield550_s3_all_0104.xml"
# coord_arr = get_grid_coordination(xml_path,n_grids_x=3,n_grids_y=3)
# np.save("grid_s3_9.npy",coord_arr)

coord_arr = get_grid_coordination(xml_path,n_grids_x=5,n_grids_y=5)
np.save("coordination_outputs/grid_s3_25.npy", coord_arr)

coord_arr = get_grid_coordination(xml_path,n_grids_x=8,n_grids_y=8)
np.save("coordination_outputs/grid_s3_64.npy", coord_arr)



