import sys, os
from TimeSeriesAnalysis.data_preprocessing.preprocess_data_chunks import preprocess_data
import TimeSeriesAnalysis.params as params

if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    s_run = consts.s_runs[sys.argv[3]]
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1])

    # loop
    preprocess_data(n_tasks=n_tasks, job_id=job_id, s_run=s_run, modality=modality,
                    win_size=params.window_size, local_den=params.local_density,
                    diff_win=params.diff_window, con_win=params.con_window,
                    track_len=params.tracks_len)
