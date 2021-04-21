import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from scipy import stats

coord_arr = np.load("coordination_outputs/grid_s3_25.npy")

for i in range(len(coord_arr)):
    fig, ax = plt.subplots()
    # im = ax.imshow(coord_arr[i]-np.mean(coord_arr[i]))
    z_score = stats.zscore(coord_arr[i])
    plt.pcolor(z_score, cmap=plt.cm.Oranges, vmin=-3, vmax=3)
    plt.colorbar()
    plt.title("Coordination grid- diff, time = {}".format(i))
    plt.savefig("coordination_outputs/grid_s3_25/{}".format(i))

size = (0, 0)
img_array = []
for filename in glob.glob(r'coordination_outputs\grid_s3_25\*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('grid_s3_25.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
