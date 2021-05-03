# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from skimage import io
from DataPreprocessing.load_tracks_xml import load_tracks_xml
import matplotlib.pyplot as plt
import numpy as np
import os


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


tm_xml = r"data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s3_all_pixelratio1.xml"
bf_video = r"C:\Users\Amit\Desktop\Experiment1_w2Brightfield_s3_all.tif"
# Load the tracks XML (TrackMate's output)
tracks01, df = load_tracks_xml(tm_xml)
im = io.imread(bf_video)

# Iterate over all frames- of the tracked cells and their brightfeild's matching image
image_size = 32


def on_edge(x, y, image_size, frame_len):
    if x + image_size >= frame_len or y + image_size >= frame_len \
            or x - image_size < 0 or y - image_size < 0:
        return True
    else:
        return False


def more_than_1_cell(i, df):
    current_time = df.iloc[i]["t_stamp"]
    current_label = df.iloc[i]["label"]
    # clean multiple cells in a crop
    n_cells = df[(df["t_stamp"] == current_time) &
                 (df["label"] != current_label) &
                 (df["x"] > x - image_size) & (df["x"] < x + image_size) &
                 (df["y"] > y - image_size) & (df["y"] < y + image_size)]["label"].nunique()
    if n_cells > 1:
        return True


for track in tracks01:
    track_ratios = []
    if len(track) < 10:
        continue
    for i in range(len(track)):
        x = int(df.iloc[i]["x"])
        y = int(df.iloc[i]["y"])
        if on_edge(x, y, image_size, im.shape[1]):
            continue
        if more_than_1_cell(i, df):
            continue
        single_cell_crop = im[int(df.iloc[i]["t_stamp"]), y - image_size:y + image_size, x - image_size:x + image_size]
        cv2.imwrite("single_cell_crop.tif", single_cell_crop)
        image = cv2.imread("single_cell_crop.tif", cv2.IMREAD_COLOR)
        blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 9), 0)  # 9,9
        mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((4, 4), np.uint8)
        image[mask_otsu != 0] = (255, 255, 255)
        #
        # plt.imshow(image)
        # plt.show()

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(image, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)
        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

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
            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            box = perspective.order_points(box)
            # cv2.drawContours(display_image, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
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
            # compute it as the ratio of pixels to supplied metric
            # (in this case, um)
            if pixelsPerMetric is None:
                pixelsPerMetric = dB / width

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric

            track_ratios.append(dimA / dimB)
    plt.plot(range(len(track_ratios)), track_ratios)
    plt.show()
