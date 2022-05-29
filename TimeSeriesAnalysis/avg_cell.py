import pickle
import numpy as np
import pandas as pd
import consts
import diff_tracker_utils as utils
from Coordination.CoordinationGraphBuilder import CoordinationGraphBuilder


def get_dir_path(to_run, con_train_n, diff_train_n):
    dir_path_score = f"30-03-2022-manual_mastodon_{to_run} local density-{local_density}, s{con_train_n}, s{diff_train_n} are train"
    second_dir = f"{diff_window} frames ERK, {con_windows} frames con track len {tracks_len}"
    utils.open_dirs(dir_path_score, second_dir)
    dir_path_score += "/" + second_dir
    return dir_path_score


if __name__ == '__main__':
    diff_window = [140, 170]
    tracks_len = 30
    con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]

    s_runs = [consts.s1, consts.s3, consts.s5, consts.s2]
    s_run = consts.s3
    path = consts.local_path
    motility = True
    intensity = False
    nuc_intensity = False
    local_density = False
    winsize = 15
    if intensity:
        to_run = "intensity"
    elif motility:
        to_run = "motility"
    elif nuc_intensity:
        to_run = "nuc_intensity"

    # get mean coordination:

    df_coord = pickle.load(
        (open(
            path + f"/Coordination/coordination_outputs/coordination_dfs/manual_tracking - only tagged tracks/coord_mastodon_s{s_run['name'][1]}.pkl",
            'rb')))
    cgb = CoordinationGraphBuilder()
    mean_coord, mean_time = cgb.read_coordination_df(df_coord)

    # load all data
    df_all, tracks_s = utils.get_tracks(path + s_run["csv_all_path"], manual_tagged_list=False)  # df_s
    df_tagged = df_all[df_all["manual"] == 1]

    # # normalise tracks before finding the average cell
    # df_normalized = pd.DataFrame()
    # for track_id, track in df_tagged.groupby('Spot track ID'):
    #     track = utils.normalize_track(track, motility, intensity, nuc_intensity, False, None)
    #     df_normalized = df_normalized.append(track)

    df_mean_cell = df_tagged.groupby('Spot frame').mean()
    df_mean_cell["Spot track ID"] = 1
    df_mean_cell['Spot frame'] = df_mean_cell.index
    df_mean_cell.index = np.arange(1, len(df_mean_cell) + 1)

    transformed_cell = utils.run_tsfresh_preprocess(df_mean_cell, s_run, motility, intensity, nuc_intensity, tracks_len,
                                                    local_density,
                                                    winsize)

    # import matplotlib.pyplot as plt
    #
    # plt.plot(df_score_cell.drop(columns="Spot track ID").T)
    # plt.show()

    pickle.dump(transformed_cell,
                open(path + f"/data/mastodon/ts_transformed_new/{to_run}/average_cell_impute_single" + s_run[
                    "name"], 'wb'))


    dir_path = get_dir_path(to_run, 1, 5)
    clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)

    cols = list(X_test.columns)
    cols.extend(["Spot frame", "Spot track ID"])
    transformed_cell = transformed_cell[cols]
    df_score_cell = utils.calc_prob(transformed_cell, clf, n_frames=260)

    df_merge = pd.merge(df_score_cell, pd.DataFrame(mean_coord[:260]), on='Spot track ID')

    print()
