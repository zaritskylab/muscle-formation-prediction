from abc import ABCMeta, abstractmethod, ABC
from skimage import io
import sys
import os

from tqdm import tqdm

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import consts
import params
from utils.data_load_save import get_tracks, downcast_df
from utils.diff_tracker_utils import *

import cv2
import numpy as np
import nuc_segmentor as segmentor
import time


class FeaturesCalculatorStrategy(object):
    """
    An abstract base class for defining models. The interface,
    to be implemented by subclasses, define standard model
    operations
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    def get_centered_image(self, ind, df, im_actin, window_size):
        x, y, spot_frame = self.get_position(ind, df)
        cropped = im_actin[spot_frame][y - window_size:y + window_size, x - window_size: x + window_size]
        return cropped

    def get_position(self, ind, df):
        x = int(df.iloc[ind]["Spot position X"] / 0.462)
        y = int(df.iloc[ind]["Spot position Y"] / 0.462)
        spot_frame = int(df.iloc[ind]["Spot frame"])
        return x, y, spot_frame

    def calc_features(self, data, vid_path, window_size, local_density):
        print("calculate features", flush=True)
        # check if a features dataframe already exist
        vid_num = os.path.basename(vid_path)[1]
        features_df_path = f"data/mastodon/features/S{vid_num}_{self.name}"
        if os.path.exists(features_df_path):
            print("data exists, returning an already built features dataframe")
            features_df = pd.read_csv(features_df_path, encoding="cp1252")
            features_df = features_df[features_df["Spot track ID"].isin(data["Spot track ID"].unique())]
            # print(features_df.head(5))
            # print(features_df.shape)
            print("data shape after calculate features: ", data.shape)
            return features_df

        im = io.imread(vid_path)
        features_df = pd.DataFrame()
        data = remove_short_tracks(data, window_size)

        for label, cell_df in tqdm(data.groupby("Spot track ID")):
            cell_features_df = self.get_single_cell_measures(label, cell_df, im, window_size)
            if (not cell_features_df.empty) and (len(cell_features_df) >= window_size):
                if local_density:
                    cell_features_df["local density"] = cell_df["local density"]

                cell_features_df["manual"] = 1
                features_df = pd.concat([features_df, cell_features_df], axis=0)

        return features_df

    @abstractmethod
    def get_single_cell_measures(self, *args):
        pass


class ActinIntensityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'actin_intensity'
        super(ActinIntensityFeaturesCalculator, self).__init__(name)

    def get_single_cell_measures(self, label, df, im_actin, window_size):
        features_df = pd.DataFrame(columns=["min", "max", "mean", "sum", "Spot track ID", "Spot frame", "x", "y"])

        for i in range(len(df)):
            img = self.get_centered_image(i, df, im_actin, window_size)
            x, y, spot_frame = self.get_position(i, df)
            try:
                min_i, max_i, mean_i, sum_i = img.min(), img.max(), img.mean(), img.sum()
                data = {"min": min_i, "max": max_i, "mean": mean_i, "sum": sum_i,
                        "Spot track ID": label, "Spot frame": spot_frame, "x": x, "y": y}
                features_df = features_df.append(data, ignore_index=True)
            except:
                continue

        return features_df


class NucleiIntensityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'nuclei_intensity'
        self.segmentor = None
        super(NucleiIntensityFeaturesCalculator, self).__init__(name)

    def activate_segmentor(self):
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

    def get_segmentation(self, cropped_img, val_location_ind):
        # _, threshold = cv2.threshold(cropped_img, 0, np.max(cropped_img), cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.activate_segmentor()
        pre_seg = self.segmentor.preprocess_image(cropped_img)
        segmented_crop = self.segmentor.segment_image(pre_seg)
        segmented_crop = self.segmentor.segment_postprocess(segmented_crop)

        value = segmented_crop[val_location_ind - 1:val_location_ind + 1,
                val_location_ind - 1:val_location_ind + 1].max()
        segmented_crop = np.where(segmented_crop == value, value, 0)

        return segmented_crop

    def get_single_cell_measures(self, label, df, im_actin, window_size):
        feature_calculator = segmentor.NucleiFeatureCalculator
        df_measures = pd.DataFrame()
        win_size_3 = window_size * 4
        missed_segmentation_counter = 0
        for i in range(len(df)):  # len(df)
            x, y, spot_frame = self.get_position(i, df)
            crop = self.get_centered_image(i, df, im_actin, win_size_3)
            try:
                segmented_crop = self.get_segmentation(crop, win_size_3)
                nuc_size = feature_calculator.size(segmented_crop, win_size_3)
                aspect_ratio = feature_calculator.aspect_ratio(segmented_crop)
            except:  # raised if image crop is empty.
                missed_segmentation_counter += 1
                continue

            data = {"nuc_size": nuc_size, "aspect_ratio": aspect_ratio, "x": x, "y": y, "Spot track ID": label,
                    "Spot frame": spot_frame}
            df_measures = df_measures.append(data, ignore_index=True)
        print(f"missed: {missed_segmentation_counter}/{len(df)}", flush=True)
        print(df_measures.shape)
        return df_measures


class MotilityFeaturesCalculator(FeaturesCalculatorStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'motility'
        super(MotilityFeaturesCalculator, self).__init__(name)

    def get_single_cell_measures(self, label, df, im_actin, window_size):
        return df


if __name__ == '__main__':
    os.chdir("/home/shakarch/muscle-formation-diff")
    # os.chdir(r'C:\Users\Amit\PycharmProjects\muscle-formation-diff')
    print("\n"
          f"===== current working directory: {os.getcwd()} =====", flush=True)
    print("running: calc_features")

    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[0]]
    # s_run = consts.s_runs['1']
    modality = "nuclei_intensity"

    feature_creator = NucleiIntensityFeaturesCalculator()
    features_df_save_path = f"/home/shakarch/muscle-formation-diff/data/mastodon/features/{s_run['name']}_{feature_creator.name}"

    print(f"\n"
          f"===== running: modality={modality}, "
          f"\nvideo={s_run['name']}, "
          f"\nlocal density={params.local_density}, "
          f"\nreg={params.registration_method}, "
          f"\nimpute func= {params.impute_func},"
          f"\nfeature_calc={feature_creator.name} =====", flush=True)

    print("\n===== loading data =====", flush=True)
    csv_path = consts.data_csv_path % (params.registration_method, s_run['name'])
    df_all, _ = get_tracks(csv_path, manual_tagged_list=True)
    df_tagged = df_all[df_all["manual"] == 1]
    del df_all


    vid_path = s_run["actin_path"] if modality == "actin_intensity" else s_run["nuc_path"]
    calculated_features = feature_creator.calc_features(df_tagged,
                                                        vid_path=vid_path,
                                                        window_size=params.window_size,
                                                        local_density=False)
    print(calculated_features)

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
