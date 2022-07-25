from abc import ABCMeta, abstractmethod, ABC
import pandas as pd
from tqdm import tqdm
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import consts as consts
import more_itertools as mit


class TSFreshTransformStrategy(object):
    '''
    An abstract base class for defining models. The interface,
    to be implemented by subclasses, define standard model
    operations
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, impute_func=impute):
        self.name = name
        self.impute_function = impute_func

    @abstractmethod
    def split_data_for_parallel_run(self, data, n_splits, current_split):
        pass

    @abstractmethod
    def ts_fresh_transform_df(self, *args):
        # Abstract method for transforming a full dataframe
        pass


class SingleCellTSFreshTransform(TSFreshTransformStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self, impute_func=impute):
        name = 'SingleCellTSFreshTransform'
        super(SingleCellTSFreshTransform, self).__init__(name, impute_func=impute_func)

    def split_data_for_parallel_run(self, data, n_splits, current_split):
        ids_chunks = split_data_by_tracks(data, n_splits)
        data_chunks = [data[data["Spot track ID"].isin(chunk)] for chunk in ids_chunks]
        return data_chunks

    def ts_fresh_transform_single_cell(self, track, target, w_size, vid_path=None):
        trans_track = pd.DataFrame()
        trans_track["Spot track ID"] = track["Spot track ID"].max()
        trans_track["target"] = target

        track_splits_lst = [track.iloc[i:i + w_size, :] for i in range(0, len(track), 1) if i < len(track) - w_size + 1]
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

    def ts_fresh_transform_df(self, df_to_transform, target, window_size, vid_path=None):
        if df_to_transform.empty:
            return pd.DataFrame()

        df_to_transform = remove_short_tracks(df_to_transform, window_size)
        tracks_list = get_tracks_list(df_to_transform, target=None)

        df_transformed = pd.DataFrame()
        print(len(tracks_list))
        for track in tqdm(tracks_list):
            track_transformed = self.ts_fresh_transform_single_cell(track, target, window_size, vid_path)
            print(track_transformed.shape)
            # self.impute_function(track_transformed)
            df_transformed = df_transformed.append(track_transformed, ignore_index=True)
        print(df_transformed.shape)

        df_transformed = self.impute_function(df_transformed)

        return df_transformed


class TimeSplitsTSFreshTransform(TSFreshTransformStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self, impute_func=impute):
        name = 'SingleCellTSFreshTransform'
        super(TimeSplitsTSFreshTransform, self).__init__(name, impute_func=impute_func)

    def split_data_for_parallel_run(self, data, n_splits, current_split):
        df_time_window_split_list = split_data_to_time_portions(data, consts.tracks_len)
        data_chunks = [list(c) for c in mit.divide(n_splits, df_time_window_split_list)]
        return data_chunks[current_split]

    def ts_fresh_transform_df(self, df_to_transform, target, window_size, to_run, vid_path=None):
        if df_to_transform.empty:
            return pd.DataFrame()

        df_to_transform = remove_short_tracks(df_to_transform, consts.tracks_len)
        df_time_window_split_list = split_data_to_time_portions(df_to_transform, consts.tracks_len)

        df_transformed = pd.DataFrame()
        for time_portion in df_time_window_split_list:

            time_portion = remove_short_tracks(time_portion, consts.tracks_len)
            if not time_portion.empty:
                portion_transformed = extract_features(time_portion,
                                                       column_id="Spot track ID",
                                                       column_sort="Spot frame",
                                                       show_warnings=False,
                                                       disable_progressbar=True,
                                                       n_jobs=8)  # 12
                # self.impute_function(portion_transformed)

                portion_transformed["Spot track ID"] = portion_transformed.index
                portion_transformed["Spot frame"] = time_portion["Spot frame"].max()
                portion_transformed["target"] = target
                df_transformed = df_transformed.append(portion_transformed, ignore_index=True)
        # if not df_transformed.empty:
        #     df_transformed = self.impute_function(df_transformed)
        return df_transformed
