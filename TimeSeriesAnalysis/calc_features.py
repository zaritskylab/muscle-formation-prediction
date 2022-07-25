from abc import ABCMeta, abstractmethod, ABC
from skimage import io
import pandas as pd
from utils.diff_tracker_utils import *
import cv2
import numpy as np
import nuc_segmentor as segmentor


class CalcFeaturesStrategy(object):
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
        cropped = im_actin[spot_frame][x - window_size:x + window_size, y - window_size: y + window_size]
        return cropped

    def get_position(self, ind, df):
        x = int(df.iloc[ind]["Spot position X (µm)"] / 0.462)
        y = int(df.iloc[ind]["Spot position Y (µm)"] / 0.462)
        spot_frame = int(df.iloc[ind]["Spot frame"])
        return x, y, spot_frame

    def calc_features(self, data, actin_vid_path, window_size, local_density):
        im = io.imread(actin_vid_path)
        features_df = pd.DataFrame()
        data = remove_short_tracks(data, window_size)

        for label, cell_df in data.groupby("Spot track ID"):
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


class ActinIntensityCalcFeatures(CalcFeaturesStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'actin_intensity'
        super(ActinIntensityCalcFeatures, self).__init__(name)

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


class NucleiIntensityCalcFeatures(CalcFeaturesStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'nuclei_intensity'
        self.segmentor = segmentor.NucSegmentor()
        super(NucleiIntensityCalcFeatures, self).__init__(name)

    def get_segmentation(self, cropped_img):
        _, threshold = cv2.threshold(cropped_img, 0, np.max(cropped_img), cv2.THRESH_TRIANGLE)
        pre_seg = self.segmentor.preprocess_image(threshold)
        segmented_crop = self.segmentor.segment_image(pre_seg)
        segmented_crop = self.segmentor.segment_postprocess(segmented_crop)

        return segmented_crop

    def get_single_cell_measures(self, label, df, im_actin, window_size):
        df_measures = pd.DataFrame(columns=["min", "max", "mean", "sum", "Spot track ID", "Spot frame", "x", "y", ])
        missed_segmentation_counter = 0
        for i in range(len(df)):  # len(df)
            x, y, spot_frame = self.get_position(i, df)
            crop = self.get_centered_image(i, df, im_actin, window_size)
            try:
                segmented_crop = self.get_segmentation(crop)
                feature_calculator = segmentor.NucleiFeatureCalculator
                nuc_size = feature_calculator.size(segmented_crop)
                min_nuc, max_nuc, mean_nuc, std_nuc, sum_nuc = feature_calculator.intensity(segmented_crop, crop)
            except:  # raised if image crop is empty.
                print(crop.shape)
                missed_segmentation_counter += 1
                continue

            data = {"nuc_size": nuc_size, "min_nuc": min_nuc, "max_nuc": max_nuc, "mean_nuc": mean_nuc, "x": x, "y": y,
                    "sum_nuc": sum_nuc, "std_nuc": std_nuc, "Spot track ID": label, "Spot frame": spot_frame}
            df_measures = df_measures.append(data, ignore_index=True)

        print(f"cell #{label}, missed spots: {missed_segmentation_counter}/{len(df)}")

        return df_measures


class MotilityCalcFeatures(CalcFeaturesStrategy, ABC):
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'motility'
        super(MotilityCalcFeatures, self).__init__(name)

    def get_single_cell_measures(self, label, df, im_actin, window_size):
        return df
