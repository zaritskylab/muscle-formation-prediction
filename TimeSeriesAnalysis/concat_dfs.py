import glob
import pickle
import os, sys

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
from params import PARAMS_DICT, impute_methodology, impute_func, registration_method
import pandas as pd
import consts
import numpy as np


def downcast_df(data):
    data_copy = data.copy()
    data_copy = data_copy.fillna(0)
    data_copy = data_copy.dropna(axis=1)
    cols = list(data_copy.drop(columns="Spot track ID").columns)
    for col in cols:
        if data_copy[col].sum().is_integer():
            data_copy[col] = pd.to_numeric(data_copy[col], downcast='integer')
        else:
            data_copy[col] = pd.to_numeric(data_copy[col], downcast='float')

        if np.isinf(data_copy[col]).sum() > 0:
            data_copy[col] = data[col]

    return data_copy


def concat_data_portions(local_density, window_size):
    print(f"running: modality={modality}, video={s_run['name']}, local density={local_density}, "
          f"reg={registration_method}, impute_methodology={impute_methodology}, impute func= {impute_func},")

    s_run_files_dir_path = path + f"/data/mastodon/ts_transformed_new/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}"
    # file_save_path = s_run_files_dir_path + \
    #                  f"/{s_run['name']}_imputed reg={registration_method},local_den={local_density}" + \
    #                  (f"win size {window_size}" if modality != "motility" else "")

    # concat all dfs
    df_all_chunks = pd.DataFrame()
    for file in glob.glob(s_run_files_dir_path + '/*.pkl'):
        if f"reg={registration_method},local_den={local_density}" in file:
            if modality != "motility":
                if f"win size {window_size}" not in file:
                    continue
            try:
                chunk_df = pickle.load(open(file, 'rb'))  # + ".pkl"
                chunk_df = downcast_df(chunk_df)
                print(chunk_df.shape)
                if chunk_df.shape[0] > 0:
                    df_all_chunks = pd.concat([df_all_chunks, chunk_df], ignore_index=True)
            except Exception as e:
                print(e)
                continue

    data_save_csv_path = s_run_files_dir_path + f"_imputed reg={registration_method}, local_den={local_density}, win size {window_size}"
    if not df_all_chunks.empty:
        pickle.dump(df_all_chunks, open(data_save_csv_path + ".pkl", 'wb'))
        print(df_all_chunks.shape)
        print("saved")
    else:
        print("directory was empty")

    try:
        loaded_csv = pickle.load(open(data_save_csv_path + ".pkl", 'rb'))

        # delete files
        print("delete files")
        for file in glob.glob(s_run_files_dir_path):
            if f"reg={registration_method},local_den={local_density}" in file:
                if modality != "motility":
                    if f"win size {window_size}" not in file:
                        continue

                os.remove(file)
    except Exception as e:
        print("could not upload csv")
        print(e)


if __name__ == '__main__':
    path = consts.cluster_path
    modality = sys.argv[1]
    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')]
    param_explore = sys.argv[2]

    PARAMS = PARAMS_DICT[param_explore]

    for exp_param in PARAMS[param_explore]:
        print(f"{param_explore} : {exp_param}")
        params_copy = PARAMS.copy()
        params_copy.update({param_explore: exp_param})

        concat_data_portions(params_copy['local_density'], params_copy['window_size'])
