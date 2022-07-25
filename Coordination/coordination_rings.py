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
    path = consts.cluster_path
    s_run_num = int(os.getenv('SLURM_ARRAY_TASK_ID')[0])
    csv_path = path + fr"/data/mastodon/all_detections_s{s_run_num}-vertices.csv"

    print(csv_path)
    save_path = path + fr"/Coordination/coordination_outputs/coordination_dfs/manual_tracking/coord_mastodon_s{s_run_num}"
    coord = CoordinationCalc(SMOOTHING_VAR=5, NEIGHBORING_DISTANCE=30, csv_path=csv_path)
    validate_change_distances(coord, save_path)
