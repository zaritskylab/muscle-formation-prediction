from abc import ABCMeta, abstractmethod, ABC
from tqdm import tqdm
import os, sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.diff_tracker_utils import *
from utils.data_load_save import *
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import more_itertools as mit


class TSFreshTransformStrategy(object):
    """
    An abstract base class for defining tsfresh transformation methods. The interface,
    to be implemented by subclasses, define the tsfresh transformation operations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, impute_func=impute):
        """
        :param name: (string) strategy name
        :param impute_func: (function) an impute function from: tsfresh.utilities.dataframe_functions
                            (impute/impute_dataframe_zero)
        """
        self.name = name
        self.impute_function = impute_func

    @abstractmethod
    def split_data_for_parallel_run(self, data, n_splits, current_split_ind, track_len) -> list:
        """
        Splits the dataset to enable parallel transformation via task arrays
        :param track_len:
        :param data: (pd.DataFrame) data to split (and process)
        :param n_splits: (int) number of needed divisions (length of the task array)
        :param current_split_ind: (int) optional.
        if not None, the function will return the current_split_ind portion of data to process, instead of the whole
        splits
        :return: (list) divided data / current split to process
        """
        pass

    @abstractmethod
    def ts_fresh_transform_df(self, df_to_transform, target, track_len) -> pd.DataFrame:
        """
        Performs tsfresh transformation for entire dataset. Each track is divided into portions in size of
        window_size. For each portion it then calculates tsfresh features.
        :param track_len: (int) length of track portions
        :param target: (int) 1 denotes "ERKi treated cell", 0 denotes "Control cell"
        :param df_to_transform: data to calculate tsfresh features from.
        :return: (pd.DataFrame) tsfresh fetures dataframe
        """
        pass


class SingleCellTSFreshTransform(TSFreshTransformStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self, impute_func=impute):
        name = 'SingleCellTSFreshTransform'
        super(SingleCellTSFreshTransform, self).__init__(name, impute_func=impute_func)

    def split_data_for_parallel_run(self, data, n_splits, current_split_ind, track_len):
        ids_chunks = split_data_by_tracks(data, n_splits)
        data_chunks = [data[data["Spot track ID"].isin(chunk)] for chunk in ids_chunks]
        return data_chunks

    @staticmethod
    def ts_fresh_transform_single_cell(track, target, track_len):
        print("transform data", flush=True)
        trans_track = pd.DataFrame()
        trans_track["Spot track ID"] = track["Spot track ID"].max()
        trans_track["target"] = target

        track_splits_lst = [track.iloc[i:i + track_len, :] for i in range(0, len(track), 1) if
                            i < len(track) - track_len + 1]
        for track_split in track_splits_lst:
            trans_split = extract_features(track_split,
                                           column_id="Spot track ID",
                                           column_sort="Spot frame",
                                           show_warnings=False,
                                           disable_progressbar=True,
                                           n_jobs=8)  # 12

            trans_split["Spot frame"] = track_split["Spot frame"].max()
            trans_track = trans_track.append(trans_split, ignore_index=True)

        return trans_track

    def ts_fresh_transform_df(self, df_to_transform, target, track_len):
        if df_to_transform.empty:
            return pd.DataFrame()

        df_to_transform = remove_short_tracks(df_to_transform, track_len)
        tracks_list = get_tracks_list(df_to_transform, target=None)

        df_transformed = pd.DataFrame()
        print(len(tracks_list))
        for track in tqdm(tracks_list):
            track_transformed = self.ts_fresh_transform_single_cell(track, target, track_len)
            print(track_transformed.shape)
            # self.impute_function(track_transformed)
            df_transformed = df_transformed.append(track_transformed, ignore_index=True)
        print(df_transformed.shape)

        df_transformed = self.impute_function(df_transformed)

        return df_transformed


class TimeSplitsTSFreshTransform(TSFreshTransformStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self, impute_func=impute):
        name = 'SingleCellTSFreshTransform'
        super(TimeSplitsTSFreshTransform, self).__init__(name, impute_func=impute_func)

    def split_data_for_parallel_run(self, data, n_splits, current_split_ind, track_len):
        df_time_window_split_list = split_data_to_time_portions(data, track_len)
        data_chunks = [list(c) for c in mit.divide(n_splits, df_time_window_split_list)]
        return data_chunks[current_split_ind]

    def ts_fresh_transform_df(self, df_to_transform, target, track_len):
        print("transform data", flush=True)
        if df_to_transform.empty:
            return pd.DataFrame()

        df_to_transform = remove_short_tracks(df_to_transform, track_len)
        df_time_window_split_list = split_data_to_time_portions(df_to_transform, track_len)

        df_transformed = pd.DataFrame()
        for time_portion in df_time_window_split_list:

            time_portion = remove_short_tracks(time_portion, track_len)
            if not time_portion.empty:
                portion_transformed = extract_features(time_portion,
                                                       column_id="Spot track ID",
                                                       column_sort="Spot frame",
                                                       show_warnings=False,
                                                       disable_progressbar=True,
                                                       n_jobs=8)  # 12

                portion_transformed["Spot track ID"] = portion_transformed.index
                portion_transformed["Spot frame"] = time_portion["Spot frame"].max()
                portion_transformed["target"] = target
                df_transformed = df_transformed.append(portion_transformed, ignore_index=True)
        return df_transformed
