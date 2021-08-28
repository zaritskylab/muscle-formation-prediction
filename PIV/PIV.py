import glob

from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import imageio
import os
from natsort import natsorted
from tqdm import tqdm


def calc_piv_2_images(frame_a, frame_b, idx, dir_name):
    '''
    Performs Particle Image Velocimetry (PIV) of two images, and saves an image with PIV on it.
    :param frame_a: first image
    :param frame_b: consecutive image
    :param idx: index of the first frame, for saving and ordering the images
    :param dir_name: directory to save the image to
    :return: -
    '''
    u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32),
                                                           frame_b.astype(np.int32),
                                                           window_size=winsize,
                                                           overlap=overlap,
                                                           dt=dt,
                                                           search_area_size=searchsize,
                                                           sig2noise_method='peak2peak')
    x, y = pyprocess.get_coordinates(image_size=frame_a.shape, search_area_size=searchsize, overlap=overlap)
    u1, v1, mask = validation.sig2noise_val(u0, v0, sig2noise, threshold=1.05)

    # to see where is a reasonable limit filter out
    # outliers that are very different from the neighbours
    u2, v2 = filters.replace_outliers(u1, v1, method='localmean', max_iter=3, kernel_size=3)

    # convert x,y to mm; convert u,v to mm/sec
    x, y, u3, v3 = scaling.uniform(x, y, u2, v2, scaling_factor=scaling_factor)  # 96.52 microns/pixel

    # 0,0 shall be bottom left, positive rotation rate is counterclockwise
    x, y, u3, v3 = tools.transform_coordinates(x, y, u3, v3)

    fig, ax = plt.subplots()
    im = np.negative(frame_a)  # plot negative of the image for more clarity
    xmax = np.amax(x) + winsize / (2 * scaling_factor)
    ymax = np.amax(y) + winsize / (2 * scaling_factor)
    ax.imshow(im, cmap="Greys_r", extent=[0.0, xmax, 0.0, ymax])

    invalid = mask.astype("bool")
    valid = ~invalid
    plt.quiver(
        x[invalid], y[invalid], u3[invalid], v3[invalid], color="r", width=width)
    plt.quiver(x[valid], y[valid], u3[valid], v3[valid], color="b", width=width)

    ax.set_aspect(1.)
    plt.title(r'Velocity Vectors Field (Frame #%d) $(\frac{\mu m}{hour})$' % idx)
    plt.savefig(dir_name + "/" + "vec_page%d.png" % idx, dpi=200)
    plt.show()
    plt.close()


def calc_piv_video(dir_name, video):
    '''
    calculates PIV for all of a video's frames, then saves the images.
    :param dir_name: directory to save images to
    :param video: TIFF file of the experiment
    :return: -
    '''
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    for i in tqdm(range(len(video) - 1)):
        frame_a = video[i]
        frame_b = video[i + 1]
        calc_piv_2_images(frame_a, frame_b, idx=i, dir_name=dir_name)


def gif_from_images(dir_name, file_name, duration=0.3):
    '''
    Creates GIF of images stored in a directory.
    :param dir_name: directory to take images from
    :param file_name: video saving name
    :param duration: ranges between 0.0 to 1.0 - determines the duration of displaying each frame
    :return: - 
    '''
    img_array = []
    files = glob.glob(dir_name + r'\*.png')

    for filename in natsorted(files):
        img = plt.imread(filename)
        img_array.append(img)
    imageio.mimsave(dir_name + "/" + file_name, img_array, duration=duration)


if __name__ == '__main__':
    # params
    winsize = 60  # 32  # pixels, interrogation window size in frame A
    searchsize = 100  # 38  # pixels, search in image B
    overlap = 40  # 12  # pixels, 50% overlap
    dt = 90  # 90  # sec, time interval between pulses
    scaling_factor = 0.462  # 1 #0.4620000 um
    width = 0.0025

    # load tiff video
    bf_video = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\18042021\70k\Nuclei\s5\s5_all.tif"
    im = io.imread(bf_video)

    dir_name = "video_outputs/PIV_s5_Nuclei"
    calc_piv_video(dir_name, im)
    gif_from_images(dir_name, "piv_video.gif", 0.3)
