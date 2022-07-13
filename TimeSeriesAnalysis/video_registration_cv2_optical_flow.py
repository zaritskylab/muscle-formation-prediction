import cv2
import numpy as np
from skimage import io

# --- Load the sequence
im_nuc = io.imread(
    r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\211212_CD7_ERK_P38\Erki\211212erki-p38-stiching_s3_tdTom_ORG.tif")
# im_nuc_new = np.zeros((im_nuc.shape))

for i in range(len(im_nuc)):
    frame1, frame2 = im_nuc[i], im_nuc[i + 1]

    prvs = frame1
    # prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # hsv = np.zeros_like(np.expand_dims(frame1, 2))
    hsv = np.zeros_like(cv2.cvtColor((np.expand_dims(frame1, 2)),cv2.COLOR_GRAY2RGB))
    hsv[..., 1] = 255

    next = frame2
    # next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)

cv2.destroyAllWindows()
