from abc import ABCMeta, abstractmethod, ABC
import sys
import os
import image_registration
import numpy as np
from pystackreg import StackReg
from skimage import io
from skimage import registration
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from skimage.transform import warp
import cv2

import consts
from tqdm import tqdm


class RegistrationStrategy(object):
    '''
    An abstract base class for defining models. The interface,
    to be implemented by subclasses, define standard model
    operations
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def calc_shifts(self, *args) -> np.array:
        pass

    @abstractmethod
    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        pass

    def register_tracks(self, flows, data_to_registrate):
        to_reg_data = data_to_registrate.copy()
        for label, label_df in tqdm(to_reg_data.groupby("Spot track ID")):

            label_df = label_df.sort_values("Spot frame")
            for i in range(0, len(label_df) - 1):

                spot_frame = label_df.iloc[i]["Spot frame"] - 1  # todo check the -1 thing
                x_pix = int(label_df.iloc[i]["Spot position X (µm)"] / 0.462)
                y_pix = int(label_df.iloc[i]["Spot position Y (µm)"] / 0.462)
                try:
                    x_flow, y_flow = self.get_reg_coordinates(flows, spot_frame, x_pix, y_pix)
                    # flow_x, flow_y = flows[spot_frame][..., 0], flows[spot_frame][..., 1]
                except:
                    print(spot_frame)
                    print(len(flows))
                    break
                x_reg = (x_pix + x_flow) * 0.462
                y_reg = (y_pix + y_flow) * 0.462

                to_reg_data.loc[to_reg_data["Spot track ID"] == label, "Spot position X (µm)"].iloc[i] = x_reg
                to_reg_data.loc[to_reg_data["Spot track ID"] == label, "Spot position Y (µm)"].iloc[i] = y_reg

        return to_reg_data


class OpticalFlowRegistration(RegistrationStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'OpticalFlowRegistration'
        super(OpticalFlowRegistration, self).__init__(name)

    def calc_shifts(self, nuclei_vid):
        flows = []
        for i in tqdm(range(len(nuclei_vid) - 2)):
            image0, image1 = nuclei_vid[i], nuclei_vid[i + 1]
            flow = cv2.calcOpticalFlowFarneback(image0, image1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)

        return flows

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix):
        flow_x, flow_y = flows[spot_frame][..., 0], flows[spot_frame][..., 1]

        return flow_x[x_pix, y_pix], flow_y[x_pix, y_pix]

    def calc_shifts_video(selfself, nuclei_vid):
        im_nuc_new = np.zeros(nuclei_vid.shape)
        for i in tqdm(range(len(im_nuc) - 2)):
            image0, image1 = im_nuc[i], im_nuc[i + 1]
            flow = cv2.calcOpticalFlowFarneback(image0, image1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # --- Compute the optical flow
            # flow = optical_flow_tvl1(image0, image1)
            flow_x, flow_y = flow[..., 0], flow[..., 1]
            # Applying flow vectors to each pixel
            height, width = image1.shape
            # Use meshgrid to Return coordinate matrices from coordinate vectors.
            # Extract row and column coordinates to which flow vector values will be added.
            row_coords, col_coords = np.meshgrid(np.arange(height), np.arange(width),
                                                 indexing='ij')  # Matrix indexing
            # For each pixel coordinate add respective flow vector to transform
            image1_warp = warp(image1, np.array([(row_coords + flow_y), (col_coords + flow_x)]),
                               mode='edge')
            im_nuc_new[i] = image1_warp
        io.imsave(path + f"/data/videos/{s_run['name']}_Nuclei_aligned.tif", im_nuc_new)


class StackRegRegistration(RegistrationStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'StackRegRegistration'
        self.sr = StackReg(StackReg.TRANSLATION)
        super(StackRegRegistration, self).__init__(name)

    def calc_shifts(self, nuclei_vid):
        transition_mats = []
        for i in tqdm(range(len(nuclei_vid) - 2)):
            image0, image1 = nuclei_vid[i], nuclei_vid[i + 1]
            transition_mat = self.sr.register(ref=image0, mov=image1)
            transition_mats.append(transition_mat)

        return transition_mats

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        flow_x, flow_y = flows[spot_frame][..., 2][0], flows[spot_frame][..., 2][1]
        return flow_x, flow_y


class CrossCorrelationRegistration(RegistrationStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'CrossCorrelationRegistration'
        super(CrossCorrelationRegistration, self).__init__(name)

    def calc_shifts(self, *args):
        shifts = []
        for i in tqdm(range(len(im_nuc) - 2)):
            image0, image1 = im_nuc[i], im_nuc[i + 1]
            shift, err, phasediff = registration.phase_cross_correlation(reference_image=image0, moving_image=image1)
            shifts.append(shift * (-1))
        return shifts

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        flow_x, flow_y = flows[spot_frame][0], flows[spot_frame][1]
        return flow_x, flow_y


class OpticalFlowTvl1Registration(RegistrationStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'OpticalFlowTvl1Registration'
        super(OpticalFlowTvl1Registration, self).__init__(name)

    def calc_shifts(self, *args):
        shifts = []
        for i in tqdm(range(len(im_nuc) - 2)):
            image0, image1 = im_nuc[i], im_nuc[i + 1]
            shift = registration.optical_flow_tvl1(reference_image=image0, moving_image=image1)
            shifts.append(shift)
        return shifts

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        flow_x, flow_y = flows[spot_frame][0], flows[spot_frame][1]
        return flow_x, flow_y


class Chi2ShiftRegistration(RegistrationStrategy, ABC):  # not working
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'Chi2ShiftRegistration'
        super(Chi2ShiftRegistration, self).__init__(name)

    def calc_shifts(self, *args):
        flows = []
        for i in tqdm(range(len(im_nuc) - 2)):
            image0, image1 = im_nuc[i], im_nuc[i + 1]
            dx, dy, edx, edy = image_registration.chi2_shift(image0, image1, upsample_factor='auto')  # shift by -1*dx
            flows.append((dx, dy))
        return flows

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        flow_x, flow_y = flows[spot_frame][0], flows[spot_frame][1]
        return flow_x, flow_y


def video_registration_factory(registrator_name):
    """Factory Method"""
    registrators = {
        "OpticalFlowRegistration": OpticalFlowRegistration(),
        # "Chi2ShiftRegistration": Chi2ShiftRegistration(),
        "StackRegRegistration": StackRegRegistration(),
        "OpticalFlowTvl1Registration": OpticalFlowTvl1Registration(),
        "CrossCorrelationRegistration": CrossCorrelationRegistration()
    }

    return registrators[registrator_name]


if __name__ == '__main__':
    path = consts.cluster_path
    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')[0]]
    reg_name = sys.argv[1]

    print(f"s_run name: {s_run['name']},"
          f"registrator name: {reg_name}")

    im_nuc = io.imread(path + s_run["nuc_path"])
    data_to_reg, _ = get_tracks(path + s_run["csv_all_path"], manual_tagged_list=False)

    registrator = video_registration_factory(reg_name)
    corrections = registrator.calc_shifts(im_nuc)
    reg_data = registrator.register_tracks(data_to_registrate=data_to_reg, flows=corrections)
    reg_data.to_csv(path + f"/data/mastodon/reg_{registrator.name}_{s_run['name']} all detections.csv")
