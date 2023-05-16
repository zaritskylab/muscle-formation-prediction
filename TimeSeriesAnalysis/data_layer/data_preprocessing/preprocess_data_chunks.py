import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-diff')
sys.path.append(os.path.abspath('../../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from configuration.params import impute_methodology, impute_func, registration_method
from data_layer.utils import *
from data_layer.data_preprocessing.data_preprocessor import data_preprocessor_factory
import pandas as pd

import warnings

warnings.filterwarnings('ignore', '.*did not have any finite values.*', )
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', pd.errors.DtypeWarning)


def preprocess_data_chunk(prep, data, vid_path, win_size, track_len, s_run):
    """
    Preprocesses a data chunk by calculating features, normalizing data, and transforming it to tsfresh features vector.
    :param prep: (object) Data preprocessor object.
    :param data: (pd.DataFrame) Data chunk to be preprocessed.
    :param vid_path: (str) Path to the video.
    :param win_size: (int) half of the size of the wanted cropped image. Default is 16.
    :param track_len: (int) length of track portions
    :param s_run: (dict) containing information about the video's data.
    :return: transformed_data: (pd.DataFrame) Transformed and preprocessed data.
    Prints:
        - Transformed data shape.
    """
    data = prep.feature_creator.calc_features(data, vid_path=vid_path, temporal_seg_size=track_len,
                                              window_size=win_size)
    preprocessed_data = prep.data_normalizer.preprocess_data(data)
    transformed_data = prep.tsfresh_transformer.ts_fresh_transform_df(df_to_transform=preprocessed_data,
                                                                      target=s_run["target"],
                                                                      track_len=track_len)
    if not transformed_data.empty:
        transformed_data = prep.imputer.impute(transformed_data)
        transformed_data = transformed_data.astype(np.float32)
        print(transformed_data.shape)

    return transformed_data


def preprocess_data(n_tasks, job_id, s_run, modality, win_size, track_len, feature_type="", specific_feature_calc=""):
    """
    Preprocesses data by transforming into tsfresh features data and saving chunks of transformed data seperately.
    :param n_tasks: (int) Number of parallel tasks.
    :param job_id: (int) Identifier of the current job.
    :param s_run: (dict) containing information about the video's data.
    :param modality:  (str) "motility"/"intensity".
    :param win_size: (int) half of the size of the wanted cropped image. Default is 16.
    :param track_len: (int) length of track portions
    :param feature_type: (str, optional) Feature type for parameter tuning.. Default is an empty string.
    :param specific_feature_type: (str, optional) Specific feature type for parameter tuning. Default is an empty string.
    :return: None
    Prints:
        - Running information, including the modality, video name, job ID, and window size.
        - Paths for saving transformed data.
        - Loading data message.
        - Progress information during preprocessing and saving.
        - Information if a data chunk is empty.
        - Exception messages if there are errors during data saving or file loading.
    """

    print(f"\nrunning: modality={modality},\nvideo={s_run['name']}, \njob_id={job_id}, \nwin_size={win_size}")

    # set paths for saving transformed data
    transformed_data_dir = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}/{feature_type}/{specific_feature_calc}"
    save_transformed_data_path = transformed_data_dir + \
                                 f"/{s_run['name']}_reg={registration_method},local_den=False" + \
                                 (f"win size {win_size}" if modality != "motility" else "")
    os.makedirs(transformed_data_dir, exist_ok=True)
    vid_path = s_run["actin_path"] if modality == "actin_intensity" else s_run["nuc_path"]

    print("\n===== loading data =====")
    tracks_csv_path = consts.data_csv_path % (registration_method, s_run['name'])
    tracks_df, _ = get_tracks(tracks_csv_path, tagged_only=True)

    # define data preprocessor, split data into chunks and take only part of them according to the job_id's value
    preprocessor = data_preprocessor_factory(modality, impute_func, impute_methodology)
    data_chunks = preprocessor.tsfresh_transformer.split_data_for_parallel_run(tracks_df, n_tasks, job_id, track_len)
    len_chunks = len(data_chunks)

    # preprocess & save each data chunk
    txt_dict = {}
    for i, data_i in enumerate(data_chunks):
        print(f"{i}/{len_chunks - 1}")
        transformed_data = preprocess_data_chunk(preprocessor, data_i, vid_path, win_size, track_len, s_run)
        if not transformed_data.empty:
            pickle.dump(transformed_data, open(save_transformed_data_path + f"{job_id}_{i}.pkl", 'wb'))
            txt_dict[f"file_{job_id}_{i}"] = f"{save_transformed_data_path}{job_id}_{i}.pkl"
        else:
            print("data chunk is empty")

        try:
            pickle.load(open(txt_dict[f"file_{job_id}_{i}"], 'rb'))
        except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
            print(f"Dataframe's file is not valid. path: {save_transformed_data_path + job_id}_{i}.csv")

    try:
        with open(f"{transformed_data_dir}/files_dict.txt", 'a') as f:
            [f.write(file_name + '\n') for file_name in txt_dict.values()]
    except Exception:
        print("cannot save txt file")


if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    s_run = consts.s_runs[sys.argv[3]]
    # s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[:1]]  #:2 for ck666 experiment

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])

    preprocess_data(n_tasks=n_tasks, job_id=task_id, s_run=s_run, modality=modality,
                    win_size=params.window_size, track_len=params.tracks_len)
