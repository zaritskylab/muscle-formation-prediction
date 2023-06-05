from abc import ABCMeta, abstractmethod, ABC
from skimage import io
import sys
import os
import pandas as pd
from tqdm import tqdm

from data_layer.utils import get_tracks, remove_short_tracks

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from configuration import consts, params

import numpy as np
# import nuc_segmentor as segmentor
import time


class FeaturesCalculatorStrategy(object):
    """
    An abstract base class for features calculation. The interface,
    to be implemented by subclasses, define standard features calculation operations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    def get_centered_image(self, ind, df, vid_arr, window_size=16) -> np.array:
        """
        The method crops an image of a single cell, were the nuclei is in the center of it.
        :param ind: (int) index of the dataframe's row were the cells position is in (time point).
        :param df: (pd.DataFrame) single cells trajectories dataframe.
        :param vid_arr: (np.array) the tiff video from which we need to take the cropped image from.
        :param window_size: (int) half of the size of the wanted cropped image. Default is 16.
        :return: (np.array) cropped image of a single cell.
        """
        x, y, spot_frame = self.get_position(ind, df)
        cropped = vid_arr[spot_frame][y - window_size:y + window_size, x - window_size: x + window_size]

        return cropped

    def get_position(self, ind, df):
        """
        Returns the position of a single cell at a given point in time
        :param ind: (int) point in time where of the needed position.
        :param df: (pd.DataFrame) single cells trajectories dataframe.
        :return: (int) x,y - position of the cell as a,
        (int) spot_frame - frame of the video where the position is in.
        """
        x = int(df.iloc[ind]["Spot position X"] / consts.PIXEL_SIZE)
        y = int(df.iloc[ind]["Spot position Y"] / consts.PIXEL_SIZE)
        spot_frame = int(df.iloc[ind]["Spot frame"])

        return x, y, spot_frame

    def calc_features(self, data, temporal_seg_size=30, window_size=16, vid_path=None):
        """
        Calculates single cell trajectories of needed measurements (features) for all trajectories, and saves it to a designated
        directory. Features are calculated using the subclass function get_single_cell_measures, ibherited by each
        modality class. :param data: (pd.DataFrame) single cells trajectories dataframe. :param temporal_seg_size:
        size of temporal segment (number of frames of the video) to calculate features upon. :param window_size: (
        int) half of the size of the wanted cropped image. Default is 16. :param vid_path: (Str) path of the video to
        crop single cell images from (for intensity features). Default is None. :param
        :return: (pd.DataFrame) single cell trajectories of needed measurements (features).
        """

        # check if a features dataframe already exist
        vid_num = os.path.basename(vid_path)[1]
        features_df_path = consts.FEATURES_DIR_PATH + f"S{vid_num}_{self.name}"
        if os.path.exists(features_df_path):
            print("data exists, returning an already built features dataframe")
            features_df = pd.read_csv(features_df_path, encoding="cp1252")
            features_df = features_df[features_df["Spot track ID"].isin(data["Spot track ID"].unique())]
            print("data shape after calculate features: ", data.shape)

        else:  # features df does not exist
            im = io.imread(vid_path)
            features_df = pd.DataFrame()
            data = remove_short_tracks(data, temporal_seg_size)

            for label, cell_df in tqdm(data.groupby("Spot track ID")):
                cell_features_df = self.get_single_cell_measures(label, cell_df, im, window_size, vid_num)
                # temporal_segment
                if (not cell_features_df.empty) and (len(cell_features_df) >= temporal_seg_size):
                    cell_features_df["manual"] = 1
                    features_df = pd.concat([features_df, cell_features_df], axis=0)

            # save the new features df
            if not features_df.empty:
                out_file = open(consts.storage_path + consts.FEATURES_DIR_PATH + "S{vid_num}_{self.name}", 'wb')
                features_df.to_csv(out_file, index=False, header=True, sep=',', encoding='utf-8')
                out_file.close()

        return features_df

    @abstractmethod
    def get_single_cell_measures(self, *args):
        """
        Calculates single cell trajectory of needed measurements (features)
        """
        pass


class ActinIntensityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    A class for actin intensity features calculation.
    """

    def __init__(self):
        name = 'actin_intensity'
        super(ActinIntensityFeaturesCalculator, self).__init__(name)

    def get_single_cell_measures(self, track_id, df, vid_arr, window_size=16, vid_num=None):
        """
        Calculates single cell trajectory of actin intensity measurements (features): min, max, mean and sum of the
        intensity within a cropped cell's image :param track_id: (int) cell's ID in the dataframe. :param df: (
        pd.DataFrame) single cells trajectories dataframe. :param vid_arr: (np.array) the tiff video from which we
        need to take the cropped image from. :param window_size: (int) half of the size of the wanted cropped image.
        Default is 16. :param vid_num: number of the video to extract images from. Videos are numbered by the stage
        number. Default is None :return: (pd.DataFrame) single cell trajectory of actin intensity features.
        """
        features_df = pd.DataFrame(columns=["min", "max", "mean", "sum", "Spot track ID", "Spot frame", "x", "y"])

        for i in range(len(df)):
            img = self.get_centered_image(i, df, vid_arr, window_size)
            x, y, spot_frame = self.get_position(i, df)
            try:
                min_i, max_i, mean_i, sum_i = img.min(), img.max(), img.mean(), img.sum()
                data = {"min": min_i, "max": max_i, "mean": mean_i, "sum": sum_i,
                        "Spot track ID": track_id, "Spot frame": spot_frame, "x": x, "y": y}
                features_df = features_df.append(data, ignore_index=True)
            except:
                continue

        return features_df


class NucleiIntensityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    A class for nuclei intensity features calculation.
    """

    def __init__(self):
        name = 'nuclei_intensity'
        self.segmentor = None
        super(NucleiIntensityFeaturesCalculator, self).__init__(name)

    def activate_segmentor(self):
        """
        Initiates a segmentor object.
        :return: None
        """
        if self.segmentor is not None:
            return
        else:
            got_segmentor = False
            while not got_segmentor:
                try:
                    self.segmentor = segmentor.NucSegmentor()
                    got_segmentor = True

                except:
                    time.sleep(10)
            return

    def get_segmentation(self, cropped_img, window_size):
        """
        Returns a segmented single cell image.
        :param cropped_img: single cell image to segment
        :param window_size: (int) half of the size of the wanted cropped image.
        Default is 16.
        :return: (np.array) segmented single cell
        """
        self.activate_segmentor()
        pre_seg = self.segmentor.preprocess_image(cropped_img)
        segmented_crop = self.segmentor.segment_image(pre_seg)
        segmented_crop = self.segmentor.segment_postprocess(segmented_crop)
        winsize_4_times = window_size * 4
        value = segmented_crop[winsize_4_times - 1:winsize_4_times + 1,
                winsize_4_times - 1:winsize_4_times + 1].max()
        segmented_crop = np.where(segmented_crop == value, value, 0)

        return segmented_crop

    def get_single_cell_measures(self, track_id, df, vid_arr, window_size, vid_num):
        """
        Calculates single cell trajectory of nuclei intensity measurements (features): nuclei size and aspect ratio
        of the nucleus. :param track_id: (int) cell's ID in the dataframe. :param df: ( pd.DataFrame) single cells
        trajectories dataframe. :param vid_arr: (np.array) the tiff video from which we need to take the cropped
        image from. :param window_size: (int) half of the size of the wanted cropped image. Default is 16. :param
        vid_num: number of the video to extract images from. Videos are numbered by the stage number. Default is None
        :return: (pd.DataFrame) single cell trajectory of actin intensity features.
        """
        feature_calculator = segmentor.NucleiFeatureCalculator
        df_measures = pd.DataFrame()
        missed_segmentation_counter = 0
        for i in range(len(df)):
            x, y, spot_frame = self.get_position(i, df)
            crop = self.get_centered_image(i, df, vid_arr, window_size)
            try:
                segmented_crop = self.get_segmentation(crop, window_size)
                nuc_size = feature_calculator.size(segmented_crop, window_size)
                aspect_ratio = feature_calculator.aspect_ratio(segmented_crop)
            except:  # raised if image crop is empty.
                missed_segmentation_counter += 1
                continue

            data = {"nuc_size": nuc_size, "aspect_ratio": aspect_ratio, "x": x, "y": y, "Spot track ID": track_id,
                    "Spot frame": spot_frame}
            df_measures = df_measures.append(data, ignore_index=True)
        print(f"missed: {missed_segmentation_counter}/{len(df)}", flush=True)
        print(df_measures.shape)
        return df_measures


class MotilityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    A class for motility features calculation.
    """

    def __init__(self):
        name = 'motility'
        super(MotilityFeaturesCalculator, self).__init__(name)

    def get_single_cell_measures(self, track_id, df, vid_arr, window_size, vid_num):
        """ No feature calcultation is needed. :param track_id: (int) cell's ID in the dataframe. :param df: ( pd.DataFrame)
        single cells trajectories dataframe. :param vid_arr: (np.array) the tiff video from which we need to take the
        cropped image from. :param window_size: (int) half of the size of the wanted cropped image. Default is 16.
        :param vid_num: (int) number of the video to extract images from. Videos are numbered by the stage number. Default
        is None. :return: (pd.DataFrame) single cell trajectory of actin intensity features. """

        return df


class LocalDensityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    A class for local density features calculation.
    """

    def __init__(self):
        name = 'local_density'
        super(LocalDensityFeaturesCalculator, self).__init__(name)
        self.neighboring_distance = 50
        self.all_tracks_df_dict = {}
        for vid_name in ["S1", "S2", "S3", "S5", "S6", "S8"]:
            self.all_tracks_df_dict[vid_name], _ = get_tracks(
                consts.data_csv_path % (params.registration_method, vid_name), manual_tagged_list=False)

    def get_local_density(self, df, x, y, t, neighboring_distance):
        neighbors = df[(np.sqrt(
            (df["Spot position X"] - x) ** 2 + (df["Spot position Y"] - y) ** 2) <= neighboring_distance) &
                       (df['Spot frame'] == t) &
                       (0 < np.sqrt((df["Spot position X"] - x) ** 2 + (df["Spot position Y"] - y) ** 2))]
        return len(neighbors)

    def get_single_cell_measures(self, track_id, tagged_df, vid_arr, window_size, vid_num):
        """
        Calculates single cell trajectory of local density measurement. Local density: the number of nuclei within a
        radius of 50 µm around the cell :param track_id: (int) cell's ID in the dataframe. :param tagged_df: (
        pd.DataFrame) single cells trajectories dataframe. :param vid_arr: (np.array) the tiff video from which we
        need to take the cropped image from. :param window_size: (int) half of the size of the wanted cropped image.
        Default is 16. :param vid_num: (int) number of the video to extract images from. Videos are numbered by the
        stage number. Default is None. :return: (pd.DataFrame) single cell trajectory of actin intensity features.
        """
        all_tracks_df = self.all_tracks_df_dict.get(f"S{vid_num}")
        if all_tracks_df is None:
            tracks_csv_path = consts.data_csv_path % (params.registration_method, f"S{vid_num}")
            all_tracks_df, _ = get_tracks(tracks_csv_path, manual_tagged_list=False)
        track = tagged_df[tagged_df["Spot track ID"] == track_id]
        spot_frames = list(track.sort_values("Spot frame")["Spot frame"])
        track_local_density = [self.get_local_density(df=all_tracks_df,
                                                      x=track[track["Spot frame"] == t]["Spot position X"].values[0],
                                                      y=track[track["Spot frame"] == t]["Spot position Y"].values[0],
                                                      t=t,
                                                      neighboring_distance=self.neighboring_distance)
                               for t in spot_frames]
        track["local density"] = track_local_density

        return track[["local density", "Spot frame", "Spot track ID"]]


if __name__ == '__main__':

    print("running: calc_features")

    s_run = consts.vid_info_dict[os.getenv('SLURM_ARRAY_TASK_ID')[0]]
    modality = "nuclei_intensity"

    feature_creator = NucleiIntensityFeaturesCalculator()
    features_df_save_path = consts.FEATURES_DIR_PATH + f"{s_run['name']}_{feature_creator.name}"

    print(f"\n== running: modality={modality}, \nvideo={s_run['name']}, "
          f"\nnreg={params.registration_method}, "
          f"\nimpute func= {params.impute_func}, \nfeature_calc={feature_creator.name} ==", flush=True)

    print("\n===== loading data =====", flush=True)
    csv_path = consts.data_csv_path % (params.registration_method, s_run['name'])
    df_all, _ = get_tracks(csv_path, manual_tagged_list=True)
    df_tagged = df_all[df_all["manual"] == 1]
    del df_all

    vid_path = s_run["actin_path"] if modality == "actin_intensity" else s_run["nuc_path"]
    calculated_features = feature_creator.calc_features(df_tagged,
                                                        vid_path=vid_path,
                                                        window_size=params.window_size)
    print(calculated_features.shape)

    if not calculated_features.empty:
        print(calculated_features.shape, flush=True)

        outfile = open(features_df_save_path, 'wb')
        calculated_features.to_csv(outfile, index=False, header=True, sep=',', encoding='utf-8')
        outfile.close()

        try:
            csv = pd.read_csv(features_df_save_path, encoding="cp1252", index_col=[0])
        except Exception as e:
            print(features_df_save_path)
            print(e)
