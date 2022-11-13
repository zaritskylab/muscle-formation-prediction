from abc import ABCMeta, abstractmethod, ABC
import sys
import os

sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import image_registration
from pystackreg import StackReg
from skimage import io
from skimage import registration
from utils.diff_tracker_utils import *
from utils.data_load_save import *
from skimage.transform import warp
import cv2
import consts
from tqdm import tqdm


class VideoRegistratorStrategy(object):
    """
    An abstract base class for defining models. The interface,
    to be implemented by subclasses, define standard model
    operations
    """
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

                spot_frame = label_df.iloc[i]["Spot frame"] - 1
                x_pix = int(label_df.iloc[i]["Spot position X"] / 0.462)
                y_pix = int(label_df.iloc[i]["Spot position Y"] / 0.462)
                try:
                    x_flow, y_flow = self.get_reg_coordinates(flows, spot_frame, x_pix, y_pix)
                except:
                    print(spot_frame)
                    print(len(flows))
                    break
                x_reg = (x_pix + x_flow) * 0.462
                y_reg = (y_pix + y_flow) * 0.462
                to_reg_data.loc[to_reg_data["Spot track ID"] == label, "Spot position X"].iloc[i] = x_reg
                to_reg_data.loc[to_reg_data["Spot track ID"] == label, "Spot position Y"].iloc[i] = y_reg

        return to_reg_data


class MeanOpticalFlowVideoRegistrator(VideoRegistratorStrategy, ABC):
    """
    I(x,y,t)=I(x+np.mean(dx),y+np.mean(dy))
    """

    def __init__(self):
        name = 'MeanOpticalFlowReg'
        super(MeanOpticalFlowVideoRegistrator, self).__init__(name)

    def calc_shifts(self, nuclei_vid):
        flows = []
        for i in tqdm(range(len(nuclei_vid) - 2)):
            image0, image1 = nuclei_vid[i], nuclei_vid[i + 1]
            flow = cv2.calcOpticalFlowFarneback(image0, image1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_x = flow[1, :, :]
            flow_y = flow[0, :, :]
            x_offset = np.mean(flow_x)
            y_offset = np.mean(flow_y)
            flows.append((x_offset, y_offset))

        return flows

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        flow_x, flow_y = flows[spot_frame][0], flows[spot_frame][1]
        return flow_x, flow_y


class OpticalFlowVideoRegistrator(VideoRegistratorStrategy, ABC):
    """
    I(x,y,t)=I(x+dx,y+dy,t+dt)
    """

    def __init__(self):
        name = 'OpticalFlowRegistration'
        super(OpticalFlowVideoRegistrator, self).__init__(name)

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
        io.imsave(f"data/videos/{s_run['name']}_Nuclei_aligned.tif", im_nuc_new)


class StackRegVideoRegistrator(VideoRegistratorStrategy, ABC):
    """
    Translation. Upon translation, a straight line is mapped to a straight line of identical orientation,
    with conservation of the distance between any pair of points. A single landmark in each image gives a complete
    description of a translation. The mapping is of the form x = u + Δu """

    def __init__(self):
        name = 'StackRegRegistration'
        self.sr = StackReg(StackReg.TRANSLATION)
        super(StackRegVideoRegistrator, self).__init__(name)

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


class CrossCorrelationVideoRegistrator(VideoRegistratorStrategy, ABC):
    """
    Efficient subpixel image translation registration by cross-correlation
    """

    def __init__(self):
        name = 'CrossCorrelationRegistration'
        super(CrossCorrelationVideoRegistrator, self).__init__(name)

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


class OpticalFlowTvl1VideoRegistrator(VideoRegistratorStrategy, ABC):
    """
    the optical flow is the vector field (u, v) verifying image1(x+u, y+v) = image0(x, y),
    where (image0, image1) is a couple of consecutive 2D frames from a sequence.
    This vector field can then be used for registration by image warping.
    """

    def __init__(self):
        name = 'OpticalFlowTvl1Registration'
        super(OpticalFlowTvl1VideoRegistrator, self).__init__(name)

    def calc_shifts(self, *args):
        shifts = []
        for i in tqdm(range(len(im_nuc) - 2)):
            image0, image1 = im_nuc[i], im_nuc[i + 1]
            shift = registration.optical_flow_tvl1(reference_image=image0, moving_image=image1)
            shifts.append(shift)
        return shifts

    def get_reg_coordinates(self, flows, spot_frame, x_pix, y_pix) -> (float, float):
        flow_x, flow_y = flows[spot_frame][0], flows[spot_frame][1]
        return flow_x[x_pix, y_pix], flow_y[x_pix, y_pix]


class Chi2ShiftVideoRegistrator(VideoRegistratorStrategy, ABC):  # not working
    """
    An ordinary least squares (OLS) linear regression model
    """

    def __init__(self):
        name = 'Chi2ShiftRegistration'
        super(Chi2ShiftVideoRegistrator, self).__init__(name)

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
        "OpticalFlowRegistration": OpticalFlowVideoRegistrator(),
        "MeanOpticalFlowReg": MeanOpticalFlowVideoRegistrator(),
        "StackRegRegistration": StackRegVideoRegistrator(),
        "OpticalFlowTvl1Registration": OpticalFlowTvl1VideoRegistrator(),
        "CrossCorrelationRegistration": CrossCorrelationVideoRegistrator()
    }

    return registrators[registrator_name]


if __name__ == '__main__':
    os.chdir("/home/shakarch/muscle-formation-diff")
    print("current working directory: ", os.getcwd())

    # print("hello")
    s_run = consts.s_runs[sys.argv[1]]
    reg_name = sys.argv[2]

    print(f"s_run name: {s_run['name']},"
          f"registrator name: {reg_name}")

    im_nuc = io.imread(s_run["nuc_path"])
    csv_path = consts.data_csv_path % ("no_reg_", s_run['name'])
    data_to_reg, _ = get_tracks(csv_path, manual_tagged_list=False)

    print("data_to_reg.shape: ", data_to_reg.shape, flush=True)
    print("data_to_reg # of tagged tracks: ", data_to_reg[data_to_reg["manual"] == 1]["Spot track ID"].nunique(),
          flush=True)

    registrator = video_registration_factory(reg_name)
    corrections = registrator.calc_shifts(im_nuc)
    reg_data = registrator.register_tracks(data_to_registrate=data_to_reg, flows=corrections)
    reg_data.to_csv(consts.storage_path + f"data/mastodon/{registrator.name}_{s_run['name']} all detections.csv")
