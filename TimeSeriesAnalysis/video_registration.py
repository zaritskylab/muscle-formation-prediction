import numpy as np
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from skimage import io
from skimage import registration
from skimage.transform import warp
import diff_tracker_utils as utils
from skimage.transform import warp
import cv2

import consts
from tqdm import tqdm


def calc_shifts(im_nuc, srun, method):
    # im_nuc_new = np.zeros((im_nuc.shape))
    flows = []

    for i in tqdm(range(len(im_nuc) - 2)):
        image0, image1 = im_nuc[i], im_nuc[i + 1]

        if method == "optical_flow":
            flow = cv2.calcOpticalFlowFarneback(image0, image1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # --- Compute the optical flow
            # flow = optical_flow_tvl1(image0, image1)
            flows.append(flow)
            flow_x, flow_y = flow[..., 0], flow[..., 1]
            # Example 2: Applying flow vectors to each pixel
            height, width = image1.shape
            # Use meshgrid to Return coordinate matrices from coordinate vectors.
            # Extract row and column coordinates to which flow vector values will be added.
            row_coords, col_coords = np.meshgrid(np.arange(height), np.arange(width),
                                                 indexing='ij')  # Matrix indexing
            # For each pixel coordinate add respective flow vector to transform
            image1_warp = warp(image1, np.array([(row_coords + flow_y), (col_coords + flow_x)]),
                               mode='edge')

        # elif method == "chi2_shift":
        #
        #     dx, dy, edx, edy = image_registration.chi2_shift(image0, image1, upsample_factor='auto')  # shift by -1*dx

        # elif method == "translation":
        #     sr = StackReg(StackReg.TRANSLATION)
        #     image1_warp = sr.register_transform(image0, image1)
        #     plt.imshow(image1_warp, cmap="gray")

        # im_nuc_new[i] = image1_warp

    # io.imsave(path + f"/data/videos/{srun['name']}_Nuclei_aligned.tif", im_nuc_new)
    np.save(f"flows{srun['name']}.npy", flows, allow_pickle=True)
    return flows


def register_tracks(srun, path):
    flows = np.load(f"flows{srun['name']}.npy")  # , allow_pickle=True
    diff_df_train, tracks_s = utils.get_tracks(path + srun["csv_all_path"], manual_tagged_list=False)
    for label, label_df in tqdm(diff_df_train.groupby("Spot track ID")):

        label_df = label_df.sort_values("Spot frame")
        for i in range(0, len(label_df) - 1):
            x = label_df.iloc[i]["Spot position X (µm)"]
            y = label_df.iloc[i]["Spot position Y (µm)"]

            x_pix = int(x / 0.462)
            y_pix = int(y / 0.462)

            spot_frame = label_df.iloc[i]["Spot frame"] - 1
            try:
                flow_x, flow_y = flows[spot_frame][..., 0], flows[spot_frame][..., 1]
            except:
                print(spot_frame)
                print(len(flows))
                break
            x_reg = (x_pix + flow_x[y_pix][x_pix]) * 0.462
            y_reg = (y_pix + flow_y[y_pix][x_pix]) * 0.462

            diff_df_train.loc[diff_df_train["Spot track ID"] == label, "Spot position X (µm)"].iloc[i] = x_reg
            diff_df_train.loc[diff_df_train["Spot track ID"] == label, "Spot position Y (µm)"].iloc[i] = y_reg

    diff_df_train.to_csv(path + f"/data/mastodon/reg_{srun['name']} all detections.csv")


if __name__ == '__main__':
    srun = consts.s5

    path = consts.cluster_path

    vid_path = path + srun["nuc_path"]
    im_nuc = io.imread(vid_path)
    method = "optical_flow"

    print(srun["name"])

    calc_shifts(im_nuc, srun, method)
    register_tracks(srun, path)
