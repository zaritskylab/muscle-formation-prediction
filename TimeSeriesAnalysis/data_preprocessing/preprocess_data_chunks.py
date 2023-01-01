import json
import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-diff')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import consts
from params import impute_methodology, impute_func, registration_method
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from data_preprocessing.data_preprocessor import data_preprocessor_factory
import pandas as pd

import warnings

warnings.filterwarnings('ignore', '.*did not have any finite values.*', )
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', pd.errors.DtypeWarning)


def preprocess_data_chunk(prep, data, vid_path, local_den, win_size, track_len, s_run) -> pd.DataFrame:
    data = prep.feature_creator.calc_features(data, vid_path=vid_path, temporal_seg=track_len,
                                              window_size=win_size, local_density=local_den)
    preprocessed_data = prep.data_normalizer.preprocess_data(data)
    transformed_data = prep.tsfresh_transformer.ts_fresh_transform_df(
        df_to_transform=preprocessed_data, target=s_run["target"], track_len=track_len)  # window_size=win_size,

    if not transformed_data.empty:
        transformed_data = prep.imputer.impute(transformed_data)
        transformed_data = transformed_data.astype(np.float32)
        print(transformed_data.shape)

    return transformed_data


def preprocess_data(n_tasks, job_id, s_run, modality, win_size, local_den, diff_win, con_win, track_len, feature_type, specific_feature_calc):
    print(f"\nrunning: modality={modality},\nvideo={s_run['name']},\nreg={registration_method}, \njob_id={job_id}, \nwin_size={win_size}")

    # set paths for saving transformed data
    transformed_data_dir = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}/feature_type_{feature_type}/{specific_feature_calc}"
    save_transformed_data_path = transformed_data_dir + \
                                 f"/{s_run['name']}_reg={registration_method},local_den={local_den}" + \
                                 (f"win size {win_size}" if modality != "motility" else "")
    os.makedirs(transformed_data_dir, exist_ok=True)
    vid_path = s_run["actin_path"] if modality == "actin_intensity" else s_run["nuc_path"]

    print("\n===== loading data =====")
    tracks_csv_path = consts.data_csv_path % (registration_method, s_run['name'])
    df_tagged, _ = get_tracks(tracks_csv_path, tagged_only=True)

    # define data preprocessor, split data into chunks and take only part of them according to the job_id's value
    preprocessor = data_preprocessor_factory(modality, "time_split_transform", impute_func, impute_methodology)
    print(preprocessor)
    data_chunks = preprocessor.tsfresh_transformer.split_data_for_parallel_run(df_tagged, n_tasks, job_id, track_len)
    len_chunks = len(data_chunks)

    # preprocess & save each data chunk
    txt_dict = {}
    for i, data_i in enumerate(data_chunks):
        print(f"{i}/{len_chunks - 1}")
        transformed_data = preprocess_data_chunk(preprocessor, data_i, vid_path, local_den, win_size, track_len, s_run)
        if not transformed_data.empty:
            print("data not empty")
            pickle.dump(transformed_data, open(save_transformed_data_path + f"{job_id}_{i}.pkl", 'wb'))
            txt_dict[f"file_{job_id}_{i}"] = f"{save_transformed_data_path}{job_id}_{i}.pkl"
        else:
            print("data is empty")

        try:
            # todo convert to note by reut
            # pickle.load(open(save_transformed_data_path + f"{job_id}_{i}.pkl", 'rb'))

            pickle.load(open(txt_dict[f"file_{job_id}_{i}"], 'rb'))
        except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
            print(f"Dataframe's file is not valid. path: {save_transformed_data_path + job_id}_{i}.csv")

    try:
        with open(f"{transformed_data_dir}/files_dict.txt", 'a') as f:
            [f.write(file_name+'\n') for file_name in txt_dict.values()]
    except Exception:
        print("cannot save txt file")


if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    s_run = consts.s_runs[sys.argv[3]]
    # s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[:1]]  #:2 for ck666 experiment

    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])

    preprocess_data(n_tasks=n_tasks, job_id=job_id, s_run=s_run, modality=modality,
                    win_size=params.window_size, local_den=params.local_density,
                    diff_win=params.diff_window, con_win=params.con_window,
                    track_len=params.tracks_len)
