import os
import sys

sys.path.append(os.path.abspath('..'))
import consts
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from data_preprocess import data_preprocessor_factory
import pandas as pd

import warnings

warnings.filterwarnings('ignore', '.*did not have any finite values.*', )
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', pd.errors.DtypeWarning)

if __name__ == '__main__':
    # path = consts.cluster_path
    path = ""
    os.chdir(os.getcwd() + r'/muscle-formation-diff')
    print("current working directory: ", os.getcwd())

    modality = sys.argv[1]
    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[0]]
    registration_method = sys.argv[2]
    job_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])
    n_tasks = int(sys.argv[3])
    impute_func = sys.argv[4]
    impute_methodology = sys.argv[5]

    preprocessor = data_preprocessor_factory(modality, "time_split_transform", impute_func, impute_methodology)

    csv_path = consts.data_csv_path % (registration_method, s_run['name'])

    srun_files_dir_path = path + f"data/mastodon/ts_transformed_new/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}"

    os.makedirs(srun_files_dir_path, exist_ok=True)
    file_save_path = srun_files_dir_path + \
                     f"/{s_run['name']}_imputed reg={registration_method},local_den={consts.local_density}" + \
                     (f"win size {consts.window_size}" if modality != "motility" else "")

    print(f"running: modality={modality}, "
          f"video={s_run['name']}, "
          f"local density={consts.local_density}, "
          f"reg={registration_method}, "
          f"job_id={job_id},"
          f"impute func= {impute_func},"
          f"impute methodology= {preprocessor.imputer.name},"
          f"feature_calc={preprocessor.feature_creator.name},"
          f"dat_preprocessor={preprocessor.data_normalizer.name},"
          f"ts_transformer={preprocessor.tsfresh_transformer.name}")

    # load data for preprocessing
    df_all, _ = get_tracks(path + csv_path, manual_tagged_list=True)
    df_tagged = df_all[df_all["manual"] == 1]
    del df_all

    data_chunks = preprocessor.tsfresh_transformer.split_data_for_parallel_run(df_tagged, n_tasks, job_id)
    len_chunks = len(data_chunks)
    for i, data_i in enumerate(data_chunks):  # todo: change to - enumerate(data_chunks):
        print(f"{i}/{len_chunks}")
        data = preprocessor.feature_creator.calc_features(data_i,
                                                          actin_vid_path=path + s_run["actin_path"],
                                                          window_size=consts.window_size,
                                                          local_density=False)
        preprocessed_data = preprocessor.data_normalizer.preprocess_data(data)
        transformed_data = preprocessor.tsfresh_transformer.ts_fresh_transform_df(df_to_transform=preprocessed_data,
                                                                                  target=s_run["target"],
                                                                                  to_run=modality,
                                                                                  window_size=consts.window_size,
                                                                                  vid_path=None)
        if not transformed_data.empty:
            transformed_data = preprocessor.imputer.impute(transformed_data)
            print(transformed_data.shape)
            print(np.sum(transformed_data.isna().sum()))

            outfile = open(file_save_path + f"{job_id}_{i}.csv", 'wb')
            transformed_data.to_csv(outfile, index=False, header=True, sep=',', encoding='utf-8')
            outfile.close()

            try:
                csv = pd.read_csv(file_save_path + f"{job_id}_{i}.csv", encoding="cp1252", index_col=[0])
            except:
                try:
                    csv = pd.read_csv(file_save_path + f"{job_id}_{i}.csv", encoding="cp1252", index_col=[0])
                except Exception as e:
                    print(file_save_path + f"{job_id}_{i}.csv")
                    print(e)
