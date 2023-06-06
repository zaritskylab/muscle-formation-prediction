import os
import sys

sys.path.append('/sise/home/reutme/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from configuration.consts import IMPUTE_METHOD, IMPUTE_FUNC, REG_METHOD
from data_layer.utils import *
from data_preprocessor import data_preprocessor_factory
import pandas as pd

import warnings

warnings.filterwarnings('ignore', '.*did not have any finite values.*', )
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', pd.errors.DtypeWarning)


def prepare_data(prep, data, vid_path, win_size, track_len, target):
    """
    Preprocesses a data chunk by calculating features, normalizing data, and transforming it to tsfresh features vector.
    First, the method calculates motility/actin features to create single cell time series. Secondly, it normalizes the
    calculated the data. Lastly, the method performs tsfresh feature extraction.
    :param prep: (object) Data preprocessor object.
    :param data: (pd.DataFrame) Data chunk to be preprocessed.
    :param vid_path: (str) Path to the video.
    :param win_size: (int) half of the size of the wanted cropped image. Default is 16.
    :param track_len: (int) length of track portions
    :param target: (int) 1 "differentiated" class; 0 for "undifferentiated" class.
    :return: transformed_data: (pd.DataFrame) Transformed and preprocessed data.
    Prints:
        - Transformed data shape.
    """
    print(vid_path)
    data = prep.feature_creator.calc_features(data=data, temporal_seg_size=track_len, window_size=win_size, vid_path=vid_path)
    preprocessed_data = prep.data_normalizer.preprocess_data(data)
    transformed_data = prep.tsfresh_transformer.ts_fresh_transform_df(preprocessed_data, target, track_len)
    if not transformed_data.empty:
        transformed_data = prep.imputer.impute(transformed_data)
    transformed_data = transformed_data.astype(np.float32)
    print(transformed_data.shape)

    return transformed_data


def prepare_data_in_parallel_chunks(tracks_df, vid_path, modality, target, n_tasks, job_id, win_size, segment_length,
                                    save_data_dir_path, save_data_path):
    """
    Prepares and saves data in parallel chunks. It transforms data into tsfresh features data and saving chunks of
    transformed data seperately.
    :param tracks_df: (DataFrame) DataFrame containing the tracks data.
    :param vid_path: (str) Path to the video.
    :param modality: (str) "motility"/"intensity".
    :param target: (int) 1 "differentiated" class; 0 for "undifferentiated" class.
    :param n_tasks: (int) Number of parallel tasks.
    :param job_id: (int) Identifier of the current job (3 digit format. for example: 101).
    :param win_size: (int) half of the size of the wanted cropped image. Default is 16.
    :param segment_length: (int) length of track portions
    :param save_data_dir_path: (str) Directory path to save the data chunks.
    :param save_data_path: (str) Base path for saving the data chunks.
    :return: None
    """
    print(vid_path)
    # define data preprocessor, split data into chunks and take only part of them according to the job_id's value
    preprocessor = data_preprocessor_factory(modality)
    data_chunks = preprocessor.tsfresh_transformer.split_data_for_parallel_run(tracks_df, n_tasks, job_id,
                                                                               segment_length)
    len_chunks = len(data_chunks)

    # preprocess & save each data chunk
    txt_dict = {}
    for i, data_i in enumerate(data_chunks):
        print(f"{i}/{len_chunks - 1}")
        data_chunk_path = f"{save_data_path}{job_id}_{i}.pkl"
        transformed_data = prepare_data(preprocessor, data_i, vid_path, win_size, segment_length, target)
        if not transformed_data.empty:
            pickle.dump(transformed_data, open(data_chunk_path, 'wb'))
            txt_dict[f"file_{job_id}_{i}"] = data_chunk_path

            try:
                pickle.load(open(txt_dict[f"file_{job_id}_{i}"], 'rb'))
            except (IOError, OSError, pickle.PickleError, pickle.UnpicklingError):
                print(f"Dataframe's file is not valid. path: {data_chunk_path}")
    try:
        with open(f"{save_data_dir_path}/files_dict.txt", 'a') as f:
            [f.write(file_name + '\n') for file_name in txt_dict.values()]
    except:
        print("cannot save txt file")


if __name__ == '__main__':
    print("running prepare_data")

    # get arguments from sbatch
    modality = sys.argv[1]
    n_tasks = int(sys.argv[2])
    vid_info = consts.vid_info_dict[sys.argv[3]]

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID')[1:])

    save_data_dir_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{IMPUTE_METHOD}_{IMPUTE_FUNC}/{vid_info['name']}"
    os.makedirs(save_data_dir_path, exist_ok=True)

    vid_path = vid_info["actin_path"] if modality == "actin_intensity" else vid_info["nuc_path"]

    # set paths for saving transformed data
    save_data_path = save_data_dir_path + \
                     f"/{vid_info['name']}_reg={REG_METHOD},local_den=False" + \
                     (f"win size {consts.WIN_SIZE}" if modality != "motility" else "")

    print("\n===== load data =====")
    tracks_csv_path = consts.data_csv_path % (REG_METHOD, vid_info['name'])
    tracks_df, _ = get_tracks(tracks_csv_path, tagged_only=True)

    prepare_data_in_parallel_chunks(tracks_df=tracks_df, vid_path=vid_path, modality=modality,
                                    target=vid_info["target"],
                                    n_tasks=n_tasks, job_id=task_id, win_size=consts.WIN_SIZE,
                                    segment_length=consts.SEGMENT_LEN, save_data_dir_path=save_data_dir_path,
                                    save_data_path=save_data_path)
