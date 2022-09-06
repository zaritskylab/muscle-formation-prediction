# import consts
import sys, os

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
from TimeSeriesAnalysis.params import PARAMS_DICT, impute_methodology, impute_func, registration_method
from TimeSeriesAnalysis.utils.diff_tracker_utils import *
from TimeSeriesAnalysis.utils.data_load_save import *
import matplotlib.pyplot as plt
from sklearn import metrics
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

from TimeSeriesAnalysis.build_models_on_transformed_tracks import load_tsfresh_csv, clean_redundant_columns


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
        # x, y = zip(*auc_scores)
        # plt.plot(x, y, label=label)

    plt.axhline(0.5, color='gray', linestyle='dashed')
    plt.xlabel("time (h)")
    plt.ylabel("auc")
    plt.title("auc over time")
    plt.legend()
    if path:
        plt.savefig(path)
    plt.show()
    plt.clf()


def load_csv_in_portions(path, modality, vid_num, registration="no_reg_", local_density=False,
                         impute_func="impute", window_size=16,
                         impute_methodology="ImputeAllData"):
    path_prefix = path + f"/data/mastodon/ts_transformed_new/{modality}/{impute_methodology}_{impute_func}/"
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

    local_density = False
    window_size = 16
    tracks_len = 30

    for con_train_n, diff_train_n, con_test_n, diff_test_n in [(1, 5, 2, 3), (2, 3, 1, 5), ]:
        print(f"con_train_n {con_train_n}, diff_train_n {diff_train_n}, "
              f"con_test_n {con_test_n}, diff_test_n {diff_test_n}", flush=True)

        dir_path = f"/home/shakarch/30-07-2022-{modality} local dens-{local_density}, s{con_train_n}, s{diff_train_n} train" + (
            f" win size {window_size}" if modality != "motility" else "")
        second_dir = f"track len {tracks_len}, impute_func-{impute_methodology}_{impute_func} reg {registration_method}"
        dir_path += "/" + second_dir

        clf, _, X_test, _, _ = load_data(dir_path, load_clf=True, load_x_train=False, load_x_test=True,
                                         load_y_test=False, load_y_train=False)

        cols = list(X_test.columns)
        del X_test

        cols = [re.sub('[^A-Za-z0-9 _]+', '', col) for col in cols]
        cols = [col.replace("Spot position X m", "Spot position X") for col in cols]
        cols = [col.replace("Spot position Y m", "Spot position Y") for col in cols]
        if "Spot track ID" not in cols:
            cols.extend(["Spot track ID"])
        if "Spot frame" not in cols:
            cols.extend(["Spot frame"])

        for vid_num in ["1 ck666", "4 ck666", "5 ck666"]:  # 6, 8, 3, 5, 1, 2,
            print(f"loading vid {vid_num}", flush=True)
            # df_s = load_csv_in_portions("/home/shakarch/muscle-formation-diff", modality, vid_num,
            #                             registration=consts.registration_method,
            #                             local_density=consts.local_density, impute_func=consts.impute_func,
            #                             impute_methodology=consts.impute_methodology)

            tsfresh_transform_path = f"/home/shakarch/muscle-formation-diff/data/mastodon/ts_transformed_new/{modality}/{impute_methodology}_{impute_func}/S{vid_num}_imputed reg=" \
                                     f"{registration_method}, local_den={local_density}, win size {window_size}.pkl"

            df_s = pickle.load(open(tsfresh_transform_path, 'rb'))

            print(f"calc avg prob vid {vid_num}", flush=True)
            df_s = df_s.rename(columns=lambda x: re.sub('[^A-Za-z0-9 _]+', '', x))

            df_score = calc_prob(df_s.loc[:, ~df_s.columns.duplicated()][cols], clf, n_frames=260)
            pickle.dump(df_score, open(dir_path + f"/df_prob_w={tracks_len}, video_num={vid_num}", 'wb'))
