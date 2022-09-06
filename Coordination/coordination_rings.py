"""
Created on Sun Apr 26 11:30:35 2020
Calculating cos of theta- the angle between the velocity vectors of each cell and it's neighbors.
Constructing a DataFrame called "coordination_outputs" for a later use.
@author: Oron (Amit)
"""
import os
import sys

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
import TimeSeriesAnalysis.consts as consts

from Coordination.CoordinationCalc_mastodon import CoordinationCalc

import numpy as np

np.seterr(divide='ignore', invalid='ignore')
import warnings

warnings.simplefilter('ignore', np.RankWarning)


def validate_change_distances(coord_calculator, save_path):
    for n_dist in (30, 100, 170, 240, 310, 380, 450, 520, 590):
        print(f"distance: {n_dist}")
        coord_calculator.NEIGHBORING_DISTANCE = n_dist
        coord_calculator.build_coordination_df(validation=False, rings=True, only_tagged=False)
        coord_calculator.save_coordinationDF(save_path + f"_n_dist={n_dist}.pkl")


if __name__ == '__main__':
    print("coordination Calculator")
    # path = consts.cluster_path
    path = ""
    os.chdir("/home/shakarch/muscle-formation-diff")
    # os.chdir(r'C:\Users\Amit\PycharmProjects\muscle-formation-diff')
    print("\n"
          f"===== current working directory: {os.getcwd()} =====")

    s_run = consts.s_runs[sys.argv[1]]
    registration_method = sys.argv[2]
    ring_size = int(sys.argv[3])
    only_tagged = sys.argv[4] == "True"
    SMOOTHING_VAR = int(sys.argv[5])
    neighboring_distance = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    print(f"running coordination rings"
          f"\nsrun={s_run['name']}, "
          f"registration_method={registration_method}, "
          f"ring_size={ring_size}, "
          f"only_tagged={only_tagged}, "
          f"neighboring_distance={neighboring_distance}", flush=True)

    csv_path = path + consts.data_csv_path % (registration_method, s_run['name'])

    print(csv_path)
    # print("ring_size: ", ring_size)
    manual_tracking_dir_path = fr"Coordination/coordination_outputs/coordination_dfs/manual_tracking"
    save_dir = os.path.join(manual_tracking_dir_path, f"ring_size_{ring_size}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + fr"/coord_mastodon_{s_run['name']} reg {registration_method}"

    coord_calculator = CoordinationCalc(SMOOTHING_VAR=SMOOTHING_VAR, NEIGHBORING_DISTANCE=neighboring_distance, csv_path=csv_path)
    coord_calculator.build_coordination_df(validation=False, rings=True, only_tagged=only_tagged, ring_size=ring_size)
    coord_calculator.save_coordinationDF(save_path + f"_n_dist={neighboring_distance}" + (
        " only tagged" if only_tagged is True else "") + f" smooth {SMOOTHING_VAR}.pkl")  # only tagged
