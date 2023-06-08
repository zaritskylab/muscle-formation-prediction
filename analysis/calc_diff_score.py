import sys, os

from model_layer.build_model import clean_redundant_columns
from model_layer.build_model import calc_state_trajectory

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))

sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from configuration.consts import IMPUTE_METHOD, IMPUTE_FUNC
from data_layer.utils import *
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


def load_csv_in_portions(path, modality, vid_num, registration="no_reg_", local_density=False,
                         impute_func="impute", window_size=16, impute_methodology="ImputeAllData"):
    """
        Loads a CSV file in portions, applying specific preprocessing steps.

        :param path: (str) The path to the directory containing the CSV file.
        :param modality: (str) The modality of the data.
        :param vid_num: (int) The video number.
        :param registration: (str) The registration method applied to the data (default: "no_reg_").
        :param local_density: (bool) Whether local density data is included (default: False).
        :param impute_func: (str) The imputation function to use (default: "impute").
        :param window_size: (int) The window size for imputation (default: 16).
        :param impute_methodology: (str) The methodology for imputation (default: "ImputeAllData").
        :return: (pd.DataFrame) The loaded data.
        """
    path_prefix = path + f"/data/mastodon/ts_transformed/{modality}/{impute_methodology}_{impute_func}/"
    end = f"_imputed reg={registration}, local_den={local_density}, win size {window_size}"

    df = pd.DataFrame()
    chunksize = 10 ** 3
    with pd.read_csv(path_prefix + f"S{vid_num}" + end, encoding="cp1252", index_col=[0],
                     chunksize=chunksize, dtype=np.float32) as reader:
        for chunk in reader:
            chunk = downcast_df(chunk)
            chunk = clean_redundant_columns(chunk)
            df = df.append(chunk)

    print(df.info(memory_usage='deep'), flush=True)
    return df


if __name__ == '__main__':
    print("diff prob auc over time bitch", flush=True)
    modality = sys.argv[1]
    s_run = consts.vid_info_dict[sys.argv[2]]
    local_density = False

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(2, 3, 1, 5), (1, 5, 2, 3)]:
        print(f"con_train_n {con_train_n}, diff_train_n {diff_train_n}, "
              f"con_test_n {con_test_n}, diff_test_n {diff_test_n}", flush=True)

        print(consts)
        dir_path = consts.intensity_model_path if modality == "actin_intensity" else consts.motility_model_path
        dir_path = dir_path % (con_train_n, diff_train_n)

        clf, _, _, _, _ = load_data(dir_path, load_clf=True, load_x_train=False, load_x_test=False,
                                    load_y_test=False, load_y_train=False)

        print(f"loading vid {s_run['name']}", flush=True)
        tsfresh_file_name = f"merged_chunks_reg=MeanOpticalFlowReg_,local_den=False,win size={consts.WIN_SIZE}.pkl"
        tsfresh_dir_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{IMPUTE_METHOD}_{IMPUTE_FUNC}/{s_run['name']}/"
        tsfresh_transform_path = tsfresh_dir_path + tsfresh_file_name
        df_s = pickle.load(open(tsfresh_transform_path, 'rb'))

        print(f"calc avg prob vid {s_run['name']}", flush=True)

        cols_to_check = list(clf.feature_names_in_) + ["Spot track ID", "Spot frame"]
        df_s = df_s[cols_to_check]
        print(df_s.shape)
        df_score = calc_state_trajectory(df_s, clf, n_frames=260)
        pickle.dump(df_score, open(dir_path + f"/df_score_vid_num_{s_run['name']}.pkl", 'wb'))
