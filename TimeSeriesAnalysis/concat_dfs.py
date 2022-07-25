import pandas as pd
import os
import sys
import consts

if __name__ == '__main__':
    path = consts.cluster_path
    modality = sys.argv[1]
    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[0]]
    registration_method = sys.argv[2]
    impute_func = sys.argv[3]
    impute_methodology = sys.argv[4]

    s_run_files_dir_path = path + f"/data/mastodon/ts_transformed_new/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}"
    file_save_path = s_run_files_dir_path + \
                     f"/{s_run['name']}_imputed reg={registration_method},local_den={consts.local_density}" + \
                     (f"win size {consts.window_size}" if modality != "motility" else "")

    data_save_csv_path = s_run_files_dir_path + f"_imputed reg={registration_method}, local_den={consts.local_density}, win size {consts.window_size}"

    print(f"running: modality={modality}, "
          f"video={s_run['name']}, "
          f"local density={consts.local_density}, "
          f"reg={registration_method}, "
          f"impute func= {impute_func},")

    # concat all dfs
    df_all_chunks = pd.DataFrame()
    for file in os.listdir(s_run_files_dir_path):
        if f"reg={registration_method},local_den={consts.local_density}" in file:
            if modality != "motility":
                if f"win size {consts.window_size}" not in file:
                    continue
            try:
                chunk_df = pd.read_csv(os.path.join(s_run_files_dir_path, file), encoding="cp1252", index_col=[0])
                print(chunk_df.shape)
                if chunk_df.shape[0] > 0:
                    df_all_chunks = pd.concat([df_all_chunks, chunk_df], ignore_index=True)
            except:
                continue
    df_all_chunks.to_csv(data_save_csv_path)
    print(df_all_chunks.shape)
    print("saved")

    try:
        loaded_csv = pd.read_csv(data_save_csv_path, encoding="cp1252", index_col=[0])
        # delete files
        print("delete files")
        for file in os.listdir(s_run_files_dir_path):
            if f"reg={registration_method},local_den={consts.local_density}" in file:
                if modality != "motility":
                    if f"win size {consts.window_size}" not in file:
                        continue

                os.remove(os.path.join(s_run_files_dir_path, file))
    except Exception as e:
        print("could not upload csv")
        print(e)
