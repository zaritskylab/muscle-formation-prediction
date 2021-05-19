# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from skimage import io
from tqdm import tqdm

from DataPreprocessing.load_tracks_xml import load_tracks_xml
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle


def get_aspect_ratio(xml_path, bf_video):
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    def on_edge(x, y, image_size, frame_len):
        if x + image_size >= frame_len or y + image_size >= frame_len \
                or x - image_size < 0 or y - image_size < 0:
            return True
        else:
            return False

    def more_than_1_cell(i, track, df):
        current_time = track.iloc[i]["t_stamp"]
        current_label = track.iloc[i]["label"]
        # clean multiple cells in a crop
        n_cells = df[(df["t_stamp"] == current_time) &
                     (df["label"] != current_label) &
                     (df["x"] > x - image_size) & (df["x"] < x + image_size) &
                     (df["y"] > y - image_size) & (df["y"] < y + image_size)]["label"].nunique()
        if n_cells > 1:
            return True

    # Load the tracks XML (TrackMate's output)
    tracks01, df = load_tracks_xml(xml_path)
    im = io.imread(bf_video)

    # Iterate over all frames- of the tracked cells and their brightfeild's matching image
    image_size = 32

    ar_df = pd.DataFrame(columns=[i for i in range(0, int(df["t_stamp"].max()))])
    for id, track in enumerate(tracks01):
        track_ratios = []
        if len(track) < 10:
            continue
        for i in range(len(track)):
            x = int(track.iloc[i]["x"])
            y = int(track.iloc[i]["y"])
            if on_edge(x, y, image_size, im.shape[1]):
                continue
            if more_than_1_cell(i, track, df):
                continue
            single_cell_crop = im[int(track.iloc[i]["t_stamp"]), y - image_size:y + image_size,
                               x - image_size:x + image_size]
            cv2.imwrite("single_cell_crop.tif", single_cell_crop)
            image = cv2.imread("single_cell_crop.tif", cv2.IMREAD_COLOR)
            blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 9), 0)  # 9,9
            mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = np.ones((4, 4), np.uint8)
            image[mask_otsu != 0] = (255, 255, 255)

            # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
            edged = cv2.Canny(image, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)
            # find contours in the edge map
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if len(cnts) == 0:
                continue
            # # sort the contours from left-to-right and initialize the
            # # 'pixels per metric' calibration variable
            (cnts, _) = contours.sort_contours(cnts)
            width = 1
            pixelsPerMetric = None

            # loop over the contours individually
            for c in cnts:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < 100:
                    continue
                # compute the rotated bounding box of the contour
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                # order the points in the contour such that they appear in top-left, top-right, bottom-right,
                # and bottom-left order, then draw the outline of the rotated bounding
                box = perspective.order_points(box)

                # unpack the ordered bounding box, then compute the midpoint between the top-left and top-right coordinates,
                # followed by the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # if the pixels per metric has not been initialized, then
                # compute it as the ratio of pixels to supplied metric (in this case, um)
                if pixelsPerMetric is None:
                    pixelsPerMetric = dB / width

                # compute the size of the object
                dimA = (dA / pixelsPerMetric) + 0.00001
                dimB = (dB / pixelsPerMetric) + 0.00001

                # print(f"dimA = {dimA}")
                # print(f"dimB = {dimB}")
                # aspect_ratio = dimA / dimB if dimA > dimB else dimB / dimA
                aspect_ratio = dimA / dimB
                ar_df.loc[id, int(track.iloc[i]["t_stamp"])] = aspect_ratio

                # track_ratios.append(dimA / dimB)
    return ar_df


if __name__ == '__main__':

    for i in (3, 4, 5, 6, 11, 12):
        # xml_path = r"data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s{}_all_pixelratio1.xml".format(i)
        # bf_video = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\260520\BF\pixel ratio 1\Experiment1_w2Brightfield_s{}_all_pixelratio1.tif".format(
        #     i)
        # ar_df = get_aspect_ratio(xml_path, bf_video)
        # pickle.dump(ar_df, open("aspect_ratio_no_if{}".format(i), 'wb'))

        ar_df = pickle.load(open("aspect_ratio_no_if{}".format(i), 'rb'))

        lst_ar = []
        for col in ar_df.columns:
            col_mean = ar_df[col]
            if col == 925:
                print()
            if col <= np.max(ar_df.columns) - 3:
                col_mean = (ar_df[col] + ar_df[col + 1] + ar_df[col + 2]+ ar_df[col + 3]) / 4
            lst_ar.append(col_mean.mean())
        plt.plot(pd.DataFrame(lst_ar, columns=["aspect_ratio"]).rolling(window=5).mean())

    plt.legend([3, 4, 5, 6, 11, 12
                ], loc='upper left', title="experiment")
    plt.xticks(np.arange(0, 921, 100), labels=np.around(np.arange(0, 921, 100) * 1.5 / 60, decimals=1))
    plt.xlabel("time [h]")
    plt.ylabel("aspect ratio")
    plt.title(r'Average aspect ratio over time -diff')
    plt.grid()
    plt.show()

    # vals = [[val] if not np.isnan(val) else [] for val in ar_df.iloc[6].values]
    # import itertools
    #
    # vals = list(itertools.chain.from_iterable(vals))
    # plt.plot(vals)
    # # plt.xlim(0, 2, 0.5)
    # plt.grid()
    # plt.show()
