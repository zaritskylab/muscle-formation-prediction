import glob
import json
import pickle
import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-diff')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import params
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


def concat_data_portions(local_density, window_size, s_run, modality, feature_type):
    print(f"running: modality={modality}, video={s_run['name']}, local density={local_density}, "
          f"reg={params.registration_method}, impute_methodology={params.impute_methodology}, impute func= {params.impute_func},")

    s_run_files_dir_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{params.impute_methodology}_{params.impute_func}/{s_run['name']}/feature_type_{feature_type}/{window_size}/"

    #todo add by reut

    # concat dfs
    df_all_chunks = pd.DataFrame()
    txt_path_file = f"{s_run_files_dir_path}files_dict.txt"
    data_files = read_txt_file(txt_path_file)
    for file in data_files:
        try:
            chunk_df = pickle.load(open(file, 'rb'))  # + ".pkl"
            chunk_df = downcast_df(chunk_df)
            print(chunk_df.shape)
            if chunk_df.shape[0] > 0:
                df_all_chunks = pd.concat([df_all_chunks, chunk_df], ignore_index=True)
        except Exception as e:
            print(e)
            continue

    data_save_csv_path = s_run_files_dir_path + f"merged_chunks_reg={params.registration_method},local_den={local_density},win size={window_size}"
    if not df_all_chunks.empty:
        pickle.dump(df_all_chunks, open(data_save_csv_path + ".pkl", 'wb'))
        print(df_all_chunks.shape)
        print("saved")
    else:
        print("directory was empty")


    try:
        loaded_csv = pickle.load(open(data_save_csv_path + ".pkl", 'rb'))

        # delete files
        print("starting delete files")

        for file in data_files:
            os.remove(file)

        print("finished delete files")
        os.remove(txt_path_file)
        print("delete txt file")
    except Exception as e:
        print("could not upload csv")
        print(e)

     # concat all dfs
    #todo convert to note by reut

    # df_all_chunks = pd.DataFrame()
    # for file in glob.glob(s_run_files_dir_path + '/*.pkl'):
    #     if f"reg={params.registration_method},local_den={local_density}" in file:
    #         if modality != "motility":
    #             if f"win size {window_size}" not in file:
    #                 continue
    #         try:
    #             chunk_df = pickle.load(open(file, 'rb'))  # + ".pkl"
    #             chunk_df = downcast_df(chunk_df)
    #             print(chunk_df.shape)
    #             if chunk_df.shape[0] > 0:
    #                 df_all_chunks = pd.concat([df_all_chunks, chunk_df], ignore_index=True)
    #         except Exception as e:
    #             print(e)
    #             continue

    # data_save_csv_path = s_run_files_dir_path + f"merged_chunks_reg={params.registration_method},local_den={local_density},win size={window_size}"
    # if not df_all_chunks.empty:
    #     pickle.dump(df_all_chunks, open(data_save_csv_path + ".pkl", 'wb'))
    #     print(df_all_chunks.shape)
    #     print("saved")
    # else:
    #     print("directory was empty")

    # try:
    #     loaded_csv = pickle.load(open(data_save_csv_path + ".pkl", 'rb'))
    #
    #     # delete files
    #     print("starting delete files")
    #     for file in glob.glob(s_run_files_dir_path + "*.pkl"):
    #         if f"{s_run['name']}_reg={params.registration_method},local_den={local_density}" in file:
    #             print(f"reg={params.registration_method},local_den={local_density} are inside file")
    #             if modality != "motility":
    #                 print("modality not equal to motility")
    #                 if f"win size {window_size}" not in file:
    #                     print(f"win size {window_size} not in file")
    #                     continue
    #
    #             os.remove(file)
    #
    #     print("finished delete files")
    # except Exception as e:
    #     print("could not upload csv")
    #     print(e)

    #todo add by reut

def read_txt_file(path):
    with open(path, 'r') as f:
        data_files = f.read().splitlines()
    return data_files




if __name__ == '__main__':
    modality = sys.argv[1]
    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')]

    concat_data_portions(params.local_density, params.window_size)
