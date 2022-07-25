from deepcell.applications import NuclearSegmentation


import numpy as np
from scipy.spatial import distance as dist
from imutils import contours, perspective
import cv2
import imutils


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


class NucleiFeatureCalculator:

    @staticmethod
    def aspect_ratio(seg_img):
        """
        calculates the aspect ratio of a single nuclei
        :param seg_img: segmented image
        :return: aspect_ratio
        """
        aspect_ratio = None
        # TODO: your code here...
        # lenght/width

        gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 40, 40)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)

        box = None
        # loop over the contours individually
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the rotated bounding box of the contour
            orig = seg_img.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

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

        if dA > dB:
            aspect_ratio = dA / dB
        else:
            aspect_ratio = dB / dA

        return aspect_ratio

    @staticmethod
    def size(seg_img):
        """
        calculates the size of a single nuclei
        :param seg_img: segmented image
        :return: size
        """

        size = np.sum(seg_img == 1)

        return size

    @staticmethod
    def intensity(seg_img, original_img):
        """
        calculates min, max, mean, std and sum of the nuclei's intensity
        :param
            seg_img: segmented image
            original_img: the original image
        :return: int_min, int_max, int_mean, int_std, int_sum
        """

        list = []
        for seg, original in zip(seg_img, original_img):
            for s, o in zip(seg, original):
                if s == 1:
                    list.append(o)

        Nuc_array = np.array(list)

        Std_Nuc = np.std(Nuc_array)
        Min_Nuc = np.min(Nuc_array)
        Max_Nuc = np.max(Nuc_array)
        Mean_Nuc = np.mean(Nuc_array)
        Sum_Nuc = np.sum(Nuc_array)

        return Min_Nuc, Max_Nuc, Mean_Nuc, Std_Nuc, Sum_Nuc


class NucSegmentor:

    def __init__(self):
        self.segmentor = NuclearSegmentation()

    def preprocess_image(self, img):
        """
        Preprocessing of a single image
        :param img: input image
        :return: reshaped image (if more preprocessing is needed, add it here!)
        """
        im = np.expand_dims(img, axis=-1)
        im = np.expand_dims(im, axis=0)
        return im

    def segment_image(self, img):
        """
        Segmentation of a single image
        :param img: preprocessed image
        :return: seg_image- ndArray of the segmentation
        """
        seg_image = self.segmentor.predict(img)
        return seg_image

    def segment_postprocess(self, seg):
        return seg.reshape(seg.shape[1], seg.shape[2])
