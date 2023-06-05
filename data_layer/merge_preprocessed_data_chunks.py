import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from data_layer.utils import *


def concat_files(data_files):
    """
    Concatenates multiple files containing DataFrame chunks into a single DataFrame.
    :param data_files: (list) List of file paths.
    :return: df_all_chunks (pd.DataFrame) Concatenated DataFrame.
    Prints:
        - Exception message if an error occurs during loading or concatenation.
    """
    df_all_chunks = pd.DataFrame()
    for file in data_files:
        try:
            chunk_df = pickle.load(open(file, 'rb'))
            chunk_df = downcast_df(chunk_df)
            if chunk_df.shape[0] > 0:
                df_all_chunks = pd.concat([df_all_chunks, chunk_df], ignore_index=True)
        except Exception as e:
            print(e)
            continue

    return df_all_chunks


def delete_temporery_files(data_files, txt_path_file):
    """
    Deletes temporary files and a text file.
    :param data_files: (list) List of file paths to be deleted.
    :param txt_path_file: (str) Path of the text file to be deleted.
    :return: None
    Prints:
        - Deletion status messages.
        - Exception message if an error occurs during file deletion.
    """
    try:
        for file in data_files:
            os.remove(file)
        os.remove(txt_path_file)
        print("finished to delete files, deleting txt file")
    except Exception as e:
        print(e)


def concat_data_portions(window_size, s_run, modality, specific_feature_type="", feature_type=""):
    """
    Concatenates data portions of transformed tsfresh data from multiple files into a single DataFrame.
    :param window_size: (int) half of the size of the wanted cropped image. Default is 16.
    :param s_run: (dict) containing information about the video's data.
    :param modality: (str) "motility"/"intensity".
    :param specific_feature_type: (str, optional) Specific feature type for parameter tuning. Default is an empty string.
    :param feature_type: (str, optional) Feature type for parameter tuning.. Default is an empty string.
    :return: None
    :prints:
        - Running information, including the modality and video name.
        - Directory path where the files are located.
        - Information about deleting temporary files, including file deletion status.
    """

    def read_txt_file(path):
        with open(path, 'r') as f:
            data_files = f.read().splitlines()
        return data_files

    print(f"running: modality={modality}, video={s_run['name']}")

    s_run_files_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{params.impute_methodology}_{params.impute_func}/{s_run['name']}/{feature_type}/{specific_feature_type}/ "
    txt_path_file = f"{s_run_files_path}files_dict.txt"
    print(s_run_files_path)

    # concat dfs
    data_files = read_txt_file(txt_path_file)
    df_all_chunks = concat_files(data_files)

    # save data
    if not df_all_chunks.empty:
        data_save_csv_path = s_run_files_path + f"merged_chunks_reg={params.registration_method},local_den=False,win size={window_size}"
        pickle.dump(df_all_chunks, open(data_save_csv_path + ".pkl", 'wb'))
    else:
        print("no data to save")

    delete_temporery_files(data_files, txt_path_file)


if __name__ == '__main__':
    modality = sys.argv[1]
    s_run = consts.vid_info_dict[os.getenv('SLURM_ARRAY_TASK_ID')]

    concat_data_portions(params.window_size, s_run, modality)
