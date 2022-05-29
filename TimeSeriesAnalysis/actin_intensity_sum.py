from skimage import io
import matplotlib.pyplot as plt
import numpy as np

video_actin_path = "../data/videos/train/S1_Actin.tif"


def plot_sum_intensity(video_actin_path, label):
    im_actin = io.imread(video_actin_path)
    actin_sum = [np.nansum(im_actin[i]) for i in range(len(im_actin))]
    plt.scatter(np.arange(0, len(actin_sum)), actin_sum, label=label)


plot_sum_intensity(r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\06102021\Control\S1_nuclei.tif", "s1")
plot_sum_intensity(r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\06102021\ERK\S5_nuclei.tif", "s5")

plot_sum_intensity(
    r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\211212_CD7_ERK_P38\Control\211212erki-p38-stiching_s2_tdTom_ORG.tif",
    "s2")
plot_sum_intensity(
    r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\211212_CD7_ERK_P38\Erki\211212erki-p38-stiching_s3_tdTom_ORG.tif",
    "s3")

plt.legend()
plt.title("intensity sum - Nuclei")
plt.show()
