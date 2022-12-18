import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from TimeSeriesAnalysis.data_preprocessing.preprocess_data_chunks import preprocess_data
import consts
from utils.diff_tracker_utils import *
from utils.data_load_save import *



if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    s_run = consts.s_runs[sys.argv[3]]
    feature_calc = sys.argv[4]
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])


    for win_size in params.feature_calc_types[feature_calc]:

        print(f"start preprocess_data with window size: {win_size}")
        preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
                        win_size=win_size, local_den=params.local_density,
                        diff_win=params.diff_window, con_win=params.con_window,
                        track_len=params.tracks_len, feature_type=feature_calc, specific_feature_calc=win_size)
        print(f"finish preprocess_data with window size: {win_size}")






