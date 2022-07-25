from collections import Counter

from tsfresh import extract_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from utils.plots_functions_utils import *
import pandas as pd


class DiffTracker:

    def __init__(self, normalize, drop_columns, concat_dfs, dir_path, diff_window, con_windows) -> None:
        super().__init__()
        self.normalize = normalize
        self.drop_columns = drop_columns
        self.concat_dfs = concat_dfs
        self.dir_path = dir_path
        self.diff_window = diff_window
        self.con_windows = con_windows
        self.clf = None

    def fit_transform_tsfresh(self, X_train, y_train, X_test, column_id="Spot track ID", column_sort="Spot frame"):
        X_train = extract_relevant_features(X_train, y_train, column_id=column_id, column_sort=column_sort,
                                            show_warnings=False)
        extracted_features = extract_features(X_test, column_id=column_id, column_sort=column_sort,
                                              show_warnings=False)
        impute(extracted_features)
        X_test = extracted_features[X_train.columns]

        return X_train, X_test

    def prep_data(self, diff_df, con_df, diff_t_window, con_t_windows, added_features=None, add_time=False):
        if added_features is None:
            added_features = []
        print("concatenating control data & ERKi data")
        df = self.concat_dfs(diff_df[diff_df["manual"] == 1], con_df[con_df["manual"] == 1], diff_t_window,
                             con_t_windows)

        print(len(df[df["target"] == True]), len(df[df["target"] == False]))

        print("dropping irrelevant columns")
        df = self.drop_columns(df, added_features=added_features)

        print("normalizing data")
        df = self.normalize(df)

        if add_time:
            df["time_stamp"] = df["Spot track ID"]

        df = df.sample(frac=1).reset_index(drop=True)

        print(len(df[df["target"] == True]), len(df[df["target"] == False]))

        y = pd.Series(df['target'])
        y.index = df['Spot track ID']
        y = get_unique_indexes(y)
        df = df.drop("target", axis=1)
        return df, y

    def train(self, diff_df_train, con_df_train, diff_df_test, con_df_test, local_density, add_time=False):
        X_train, y_train = self.prep_data(diff_df=diff_df_train, con_df=con_df_train, diff_t_window=self.diff_window,
                                          con_t_windows=self.con_windows, add_time=add_time,
                                          added_features=['local density'])
        X_test, y_test = self.prep_data(diff_df=diff_df_test, con_df=con_df_test, diff_t_window=self.diff_window,
                                        con_t_windows=self.con_windows, add_time=add_time,
                                        added_features=['local density'])
        print("fit into ts-fresh dataframe")
        X_train, X_test = self.fit_transform_tsfresh(X_train, y_train, X_test)

        print("training")
        print(X_train.shape)

        clf = train_model(X_train, y_train)

        save_data(self.dir_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                  clf=clf)
        self.clf = clf
        return clf


    def evaluate_clf(self, X_test, y_test):
        # load the model & train set & test set
        clf, X_train, X_test, y_train, y_test = load_data(self.dir_path)
        report, auc_score = evaluate(clf, X_test, y_test)

        # plot ROC curve
        plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=self.dir_path)

        # perform PCA analysis
        principal_df, pca = build_pca(3, X_test)
        plot_pca(principal_df, pca, self.dir_path)

        # calculate feature importance
        plot_feature_importance(clf, X_train.columns, self.dir_path)

        # save classification report & AUC score
        txt_file = open(self.dir_path + '/info.txt', 'a')
        txt_file.write(f"classification report: {report}\n auc score: {auc_score}\n samples:{Counter(y_train)}")

        txt_file.close()
