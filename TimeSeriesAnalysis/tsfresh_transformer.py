from abc import ABCMeta, abstractmethod, ABC
import pandas as pd
from tqdm import tqdm
import diff_tracker_utils as utils
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


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
    def ts_fresh_transform_df(self, *args):
        # Abstract method for transforming a full dataframe
        pass


class SingleCellTSFreshTransform(TSFreshTransformStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'SingleCellTSFreshTransform'
        super(SingleCellTSFreshTransform, self).__init__(name)

    def ts_fresh_transform_single_cell(self, track, target, to_run, w_size, vid_path=None):
        # todo : track = normalize_track(track, motility, visual, nuc_intensity, shift_tracks, vid_path)

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

    def ts_fresh_transform_df(self, df_to_transform, target, to_run, window_size, vid_path=None):
        if df_to_transform.empty:
            return pd.DataFrame()

        df_to_transform = utils.remove_short_tracks(df_to_transform, window_size)
        tracks_list = utils.get_tracks_list(df_to_transform, target=None)

        df_transformed = pd.DataFrame()
        for track in tqdm(tracks_list):
            track_transformed = self.ts_fresh_transform_single_cell(track, target, to_run, window_size, vid_path)
            # self.impute_function(track_transformed)
            df_transformed = df_transformed.append(track_transformed, ignore_index=True)
        self.impute_function(df_transformed)

        return df_transformed


class TimeSplitsTSFreshTransform(TSFreshTransformStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'SingleCellTSFreshTransform'
        super(TimeSplitsTSFreshTransform, self).__init__(name)

    def ts_fresh_transform_df(self, df_to_transform, target, to_run, window_size, vid_path=None):
        if df_to_transform.empty:
            return pd.DataFrame()

        df_to_transform = utils.remove_short_tracks(df_to_transform, window_size)
        df_time_window_split_list = utils.split_data_to_time_portions(df_to_transform, window_size)

        df_transformed = pd.DataFrame()
        for time_portion in tqdm(df_time_window_split_list):

            time_portion = utils.remove_short_tracks(time_portion, window_size)
            if not time_portion.empty:
                # todo: time_portion = normalize_track(time_portion, motility, visual, nuc_intensity, shift_tracks, vid_path)
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

        self.impute_function(df_transformed)
        return df_transformed
