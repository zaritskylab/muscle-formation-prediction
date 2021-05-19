from DataPreprocessing.load_tracks_xml import *
from tsfresh import extract_features, extract_relevant_features, select_features
import tsfresh
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, train_test_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")

def drop_columns(df):
    intensity_features = ['median_intensity',
                          'min_intensity', 'max_intensity', 'mean_intensity',
                          'total_intensity', 'std_intensity', 'contrast', 'snr']
    basic_features = ['t_stamp', 'z', 'spot_id']
    motion_features = ['x', 'y', 'w', 'q']
    basic_features.extend(intensity_features)
    df = df[df.columns.drop(basic_features)]
    return df


def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new


def normalize_tracks(df):
    for label in df['label']:
        df[df["label"] == label]["x"]= 1
            # pd.DataFrame(df[df["label"] == label]["x"] - df[df["label"] == label].iloc[0]["x"]).values
        # df[df["label"]==label]["x"] =  labeld_df["x"]- labeld_df.iloc[0]["x"]
        # df[df["label"]==label]["y"] =  labeld_df["y"]- labeld_df.iloc[0]["y"]
    return  df



def concat_dfs(lst_videos=[2, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    max_val = 0
    total_df = pd.DataFrame()
    for i in lst_videos:  # 1,3,
        xml_path = r"data/tracks_xml/0104/Experiment1_w1Widefield550_s{}_all_0104.xml".format(i)
        _, df = load_tracks_xml(xml_path)
        df = drop_columns(df)
        df = normalize_tracks(df)
        df.label = df.label + max_val
        max_val = df["label"].max() + 1

        target = True if i in (3, 4, 5, 6, 11, 12) else False
        df['target'] = np.array([target for i in range(len(df))])
        total_df = pd.concat([total_df, df], ignore_index=True)
    return total_df


def long_extract_features(df):
    y = pd.Series(df['target'])
    y.index = df["label"]
    y = get_unique_indexes(y)
    df = df[df.columns.drop(['target'])]
    extracted_features = extract_features(df, column_id="label", column_sort="t")
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    return features_filtered


def short_extract_features(df, y):
    df = df[df.columns.drop(['target'])]
    features_filtered_direct = extract_relevant_features(df, y, column_id="label", column_sort='t', show_warnings=False,
                                                         n_jobs=8)
    return features_filtered_direct


def feature_importance(clf):
    importance = pd.Series(clf.feature_importances_)
    importance.sort_values(ascending=False)
    print(importance)

    imp_frame = importance.to_frame()
    imp_frame.plot(kind="bar")
    plt.xticks([])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance Plot')
    plt.show()


def get_single_cells_diff_scor_plot(tracks, clf):
    all_probs = []
    for cell in (5, 7, 8, 9, 10):  # , 14, 16, 19, 27, 30
        true_prob = []
        n = 20
        for i in range(0, len(tracks[cell]), n):
            x = 0
            track_portion = tracks[cell][i:i + n]
            extracted_features = extract_features(track_portion, column_id="label", column_sort="t",
                                                  show_warnings=False, n_jobs=8)
            impute(extracted_features)
            X = extracted_features[features_filtered_direct.columns]
            probs = clf.predict_proba(X)
            true_prob.append(probs[0][1])
            print(f"track portion: [{i}:{i + n}]")
            print(clf.classes_)
            print(probs)
            track_len = len(tracks[cell])
            label_time = [(tracks[cell].iloc[val]['t'] / 60) / 60 for val in range(0, track_len, n)]

            plt.plot(range(0, track_len, n), true_prob)
            plt.xticks(range(0, track_len, n), labels=np.around(label_time, decimals=1))
            plt.ylim(0, 1, 0.1)
            plt.title(f"probability to differentiation over time- diff, cell #{cell}")
            plt.xlabel("time [h]")
            plt.ylabel("prob")
            plt.grid()
            plt.show()
        all_probs.append(true_prob)
        return all_probs


def train(features_filtered, y):
    X_train, X_test, y_train, y_test = train_test_split(features_filtered, y,
                                                        test_size=0.3,
                                                        random_state=42, shuffle=True)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    # feature_importance(clf)
    predicted = cross_val_predict(clf, X_test, y_test, cv=5)
    print(classification_report(y_test, predicted))
    print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Random Forest')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig('ROC', dpi=300)
    plt.show()

    return clf


if __name__ == '__main__':
    print("start")
    df = concat_dfs([1, 3])  # 2, # 2, 4, 5, 6, 7, 8, 9, 10, 11, 12
    y = pd.Series(df['target'])
    y.index = df["label"]
    y = get_unique_indexes(y)
    pickle.dump(y, open("y_1_3_no_intensity", 'wb'))
    # y = pickle.load(open("data/y_5_12", 'rb'))

    features_filtered_direct = short_extract_features(df, y)
    pickle.dump(features_filtered_direct, open("features_filtered_no_motion_1_3", 'wb'))

    clf = train(features_filtered_direct, y)

    # features_filtered_direct = pickle.load(open("muscle-formation-diff/data/features_filtered_direct_5_12", 'rb'))
    # xml_path_diff = r"muscle-formation-diff/data/tracks_xml/0104/Experiment1_w1Widefield550_s4_all_0104.xml"
    # xml_path_con = r"muscle-formation-diff/data/tracks_xml/0104/Experiment1_w1Widefield550_s2_all_0104.xml"
    # tracks_dif, df_diff = load_tracks_xml(xml_path_diff)
    # tracks_con, df_con = load_tracks_xml(xml_path_con)
    # df_diff = drop_columns(df_diff)
    # df_con = drop_columns(df_con)
