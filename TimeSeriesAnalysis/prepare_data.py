import os
import sys

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))

import consts
from params import PARAMS_DICT, impute_methodology, impute_func, registration_method, get_tine_windows
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from data_preprocess import data_preprocessor_factory
import pandas as pd

import warnings

warnings.filterwarnings('ignore', '.*did not have any finite values.*', )
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', pd.errors.DtypeWarning)


def prepare_data(n_tasks, job_id, s_run, modality, win_size, local_den, diff_win, con_win, track_len):
    srun_files_dir_path = f"data/mastodon/ts_transformed_new/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}"
    os.makedirs(srun_files_dir_path, exist_ok=True)
    data_save_path = srun_files_dir_path + \
                     f"/{s_run['name']}_imputed reg={registration_method},local_den={PARAMS['local_density']}" + \
                     (f"win size {win_size}" if modality != "motility" else "")

    preprocessor = data_preprocessor_factory(modality, "time_split_transform", impute_func, impute_methodology)
    print(f"\n"
          f"===== running: modality={modality},\nvideo={s_run['name']}, \nlocal density={local_den}, "
          f"\nreg={registration_method}, \njob_id={job_id},\nimpute func= {impute_func},"
          f"\nimpute methodology= {preprocessor.imputer.name},\nfeature_calc={preprocessor.feature_creator.name},"
          f"\ndat_preprocessor={preprocessor.data_normalizer.name},"
          f"\nts_transformer={preprocessor.tsfresh_transformer.name} =====")

    print("\n===== loading data =====")
    tracks_csv_path = consts.data_csv_path % (registration_method, s_run['name'])
    df_all, _ = get_tracks(tracks_csv_path, manual_tagged_list=True)
    df_tagged = df_all[df_all["manual"] == 1]
    print(df_tagged.shape)
    print(df_tagged.head(3))
    del df_all

    data_chunks = preprocessor.tsfresh_transformer.split_data_for_parallel_run(df_tagged, n_tasks, job_id, track_len)
    len_chunks = len(data_chunks)
    for i, data_i in enumerate(data_chunks):
        print(f"{i}/{len_chunks}")
        vid_path = s_run["actin_path"] if modality == "actin_intensity" else s_run["nuc_path"]
        print("calculate features", flush=True)
        data = preprocessor.feature_creator.calc_features(data_i, actin_vid_path=vid_path,
                                                          window_size=win_size, local_density=local_den)
        print("data shape after calculate features: ", data.shape)
        print("normalize data", flush=True)
        preprocessed_data = preprocessor.data_normalizer.preprocess_data(data)
        print("transform data", flush=True)
        transformed_data = preprocessor.tsfresh_transformer.ts_fresh_transform_df(df_to_transform=preprocessed_data,
                                                                                  target=s_run["target"],
                                                                                  # window_size=win_size,
                                                                                  track_len=track_len)
        if not transformed_data.empty:
            transformed_data = preprocessor.imputer.impute(transformed_data)
            transformed_data = transformed_data.astype(np.float32)
            print(transformed_data.shape)
            print(np.sum(transformed_data.isna().sum()))

            pickle.dump(transformed_data, open(data_save_path + f"{job_id}_{i}.pkl", 'wb'))

            try:
                csv = pickle.load(open(data_save_path + f"{job_id}_{i}.pkl", 'rb'))
            except Exception as e:
                print(data_save_path + f"{job_id}_{i}.csv")
                print(e)


if __name__ == '__main__':
    os.chdir("/home/shakarch/muscle-formation-diff")
    # os.chdir(r'C:\Users\Amit\PycharmProjects\muscle-formation-diff')
    print("\n"
          f"===== current working directory: {os.getcwd()} =====")
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    param_explore = sys.argv[3]
    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[:1]]  #:2 for ck666 experiment
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[2])

    PARAMS = PARAMS_DICT[param_explore]

    for exp_param in PARAMS[param_explore]:
        print(f"{param_explore} : {exp_param}")
        params_copy = PARAMS.copy()
        params_copy.update({param_explore: exp_param})

        if param_explore == "tracks_len":
            diff_window, con_window = get_tine_windows(params_copy["start"], params_copy["tracks_len"])
            params_copy["diff_window"] = diff_window

            params_copy["con_window"] = con_window
        print(params_copy["window_size"])
        print(params_copy)
        prepare_data(n_tasks=n_tasks, job_id=job_id, s_run=s_run, modality=modality,
                     win_size=params_copy["window_size"], local_den=params_copy["local_density"],
                     diff_win=params_copy["diff_window"], con_win=params_copy["con_window"], track_len=params_copy["tracks_len"])
