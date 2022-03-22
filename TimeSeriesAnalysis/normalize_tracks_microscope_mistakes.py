import cv2
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cluster_path = "muscle-formation-diff"
    local_path = ".."
    path = local_path

    # diff_df_train = pd.read_csv(path + "/data/mastodon/train/Nuclei_5-vertices.csv", encoding="cp1252").dropna()
    #
    # diff_df_train = diff_df_train.sample(frac=0.01) #TODO: remove
    #
    # delta_diff = pd.DataFrame()
    #
    # for label, label_df in diff_df_train.groupby("Spot track ID"):
    #     label_df = label_df.sort_values("Spot frame")
    #     for i in range(1, len(label_df)):
    #         data_curr_frame = label_df.iloc[i-1]
    #         data_next_frame = label_df.iloc[i]
    #
    #         data_delta = data_next_frame.copy()
    #         data_delta["Spot position X (µm)"] = data_next_frame["Spot position X (µm)"] - data_curr_frame["Spot position X (µm)"]
    #         data_delta["Spot position Y (µm)"] = data_next_frame["Spot position Y (µm)"] - data_curr_frame["Spot position Y (µm)"]
    #         delta_diff = delta_diff.append(data_delta)
    #
    # for frame in range(int(delta_diff["Spot frame"].max())):
    #     avg_x = delta_diff["Spot position X (µm)"].mean()
    #     std_x = delta_diff["Spot position X (µm)"].std()
    #
    #     avg_y = delta_diff["Spot position Y (µm)"].mean()
    #     std_y = delta_diff["Spot position Y (µm)"].std()

    im_actin = io.imread(
        r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\211212_CD7_ERK_P38\Erki\211212erki-p38-stiching_s3_tdTom_ORG.tif")
    im_actin_new = np.zeros((im_actin.shape))
    for i in range(1, len(im_actin)):
        # Read the images to be aligned
        im1 = im_actin[i - 1]
        im2 = im_actin[i]

        # Convert images to grayscale
        # im1_gray = cv2.cvtColor(cv2.cvtColor(im1, cv2.COLOR_BGR2BGRA), cv2.COLOR_BGRA2GRAY)
        # im2_gray = cv2.cvtColor(cv2.cvtColor(im2, cv2.COLOR_BGR2BGRA), cv2.COLOR_BGRA2GRAY)

        im1_gray = im1
        im2_gray = im2

        # Find size of image1
        sz = im1.shape

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Specify the number of iterations.
        number_of_iterations = 5000

        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

        im_actin_new[i] = im2_aligned
        io.imsave("../data/videos/S3_Nuclei_aligned.tif", im_actin_new)

#
# import Image
#
# im__ = Image.fromarray(im_actin_new)
# im__.save("../data/videos/S3_Nuclei_aligned.tif")
