import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

coord_arr = np.load("coordination_outputs/diff_grid.npy")
for i in range(30):  # len(coord_arr)):
    fig, ax = plt.subplots()
    # im = ax.imshow(coord_arr[i]-np.mean(coord_arr[i]))
    plt.pcolor(coord_arr[i] / np.sum(coord_arr[i]), cmap=plt.cm.seismic, vmin=-0.3, vmax=0.3)
    plt.colorbar()
    plt.title("Coordination grid- diff, time = {}".format(i))
    plt.savefig("diff_grid/{}".format(i))

img_array = []
for filename in glob.glob(r'C:\Users\Amit\PycharmProjects\muscle-formation-diff\Scripts\Coordination\diff_grid\*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
