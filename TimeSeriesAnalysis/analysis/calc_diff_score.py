# import consts
import sys, os

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('../..'))

sys.path.append(os.path.abspath('../../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from configuration.params import impute_methodology, impute_func
from data_layer.utils import *
import matplotlib.pyplot as plt
from sklearn import metrics
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from model_layer import clean_redundant_columns


def auc_over_time(df_con, df_diff, clf):
    def get_t_pred(df):
        df.dropna(inplace=True)
        t_pred = []
        for track_id, track in df.groupby("Spot track ID"):
            track_to_predict = track[track["Spot frame"] == t].drop(["Spot track ID", "Spot frame"], axis=1)
            if len(track_to_predict) > 0:
                pred = clf.predict(track_to_predict)
                t_pred.append(pred[0])
        return t_pred

    time_points = list(df_con.sort_values("Spot frame")["Spot frame"].unique())
    aucs = {}
    for t in time_points:
        t_pred_con = get_t_pred(df_con)
        t_pred_diff = get_t_pred(df_diff)
        true_pred_con = [0 for i in range(len(t_pred_con))]
        true_pred_diff = [1 for i in range(len(t_pred_diff))]

        fpr, tpr, thresholds = metrics.roc_curve(true_pred_con + true_pred_diff, t_pred_con + t_pred_diff,
                                                 pos_label=1)
        aucs[t * 5 / 60] = metrics.auc(fpr, tpr)
    return aucs


def plot_auc_over_time(aucs_lst, path=None, time=(0, 25)):
    for aucs, label in aucs_lst:
        auc_scores = pd.DataFrame({"time": aucs.keys(), "auc": aucs.values()})
        auc_scores = auc_scores[(auc_scores["time"] >= time[0]) & (auc_scores["time"] <= time[1])]
        plt.plot(auc_scores["time"], auc_scores["auc"], label=label)

    plt.axhline(0.5, color='gray', linestyle='dashed')
    plt.xlabel("time (h)")
    plt.ylabel("auc")
    plt.title("auc over time")
    plt.ylim((0.2, 1))
    plt.legend()
    if path:
        plt.savefig(path, format="eps")
    plt.show()
    plt.clf()


def load_csv_in_portions(path, modality, vid_num, registration="no_reg_", local_density=False,
                         impute_func="impute", window_size=16,
                         impute_methodology="ImputeAllData"):
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
    s_run = consts.s_runs[sys.argv[2]]
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
        tsfresh_file_name = f"merged_chunks_reg=MeanOpticalFlowReg_,local_den=False,win size={params.window_size}.pkl"
        tsfresh_dir_path = consts.storage_path + f"data/mastodon/ts_transformed/{modality}/{impute_methodology}_{impute_func}/{s_run['name']}/"
        tsfresh_transform_path = tsfresh_dir_path + tsfresh_file_name
        df_s = pickle.load(open(tsfresh_transform_path, 'rb'))

        print(f"calc avg prob vid {s_run['name']}", flush=True)

        cols_to_check = list(clf.feature_names_in_) + ["Spot track ID", "Spot frame"]
        df_s = df_s[cols_to_check]
        print(df_s.shape)
        df_score = calc_prob(df_s, clf, n_frames=260)
        pickle.dump(df_score, open(dir_path + f"/df_score_vid_num_{s_run['name']}.pkl", 'wb'))

