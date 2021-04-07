import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import imageio

from Scripts.DataPreprocessing.CellsCropper import crop_cells

i = 3
bf_video = r"../../data/videos/Experiment1_w2Brightfield_s{}_all.tif".format(i)
tm_xml = r'../../data/tracks_xml/Experiment1_w1Widefield550_s{}_all.xml'.format(i)

images, _ = crop_cells(normalized=True, bf_video=bf_video, tm_xml=tm_xml,
                       resize=False, resize_to=64, image_size=64, crop_single_cell=True)

# # normalize images:
# for i in range(1, len(images)):
#     images[i] = ((images[i] - images[i].min()) * (255 - 0)) / (images[i].max() - images[i].min())

mean_intensity_vals = []
min_intensity_vals = []
max_intensity_vals = []
remainder_values = []
reminder_min = []
reminder_max = []
reminder_mean = []
remainder_sum = []
# get adherence to substrate by subtracting: frame1-frame0
for i in range(1, len(images)):
    remainder = images[i] - images[i - 1]
    remainder_values.append(remainder)
    reminder_min.append(np.min(remainder))
    reminder_max.append(np.max(remainder))
    reminder_mean.append(np.mean(remainder))
    remainder_sum.append(np.sum(remainder))

    image = images[i]
    mean_intensity_vals.append(np.mean(image))
    min_intensity_vals.append(np.min(image))
    max_intensity_vals.append(np.max(image))

plt.plot(range(1, len(images)), remainder_values)
plt.show()

plt.plot(range(1, len(images)), mean_intensity_vals)
plt.plot(range(1, len(images)), min_intensity_vals)
plt.plot(range(1, len(images)), max_intensity_vals)
plt.show()

# print(len(images))
# imageio.mimsave('movie.gif', images)

print("min", np.min(images[0]))
print("max", np.max(images[0]))
print("mean", np.mean(images[0]))
print("std", np.std(images[0]))
print(images[0])
