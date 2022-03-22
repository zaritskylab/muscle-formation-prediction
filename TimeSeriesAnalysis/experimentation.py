import os
import numpy as np
import skimage.io
from skimage import io

from tifffile import imsave
import sys

from tqdm import tqdm


def get_shifts(vid_path, n_frames_to_test):
    full_img_stack = io.imread(vid_path)

    full_img_stack = full_img_stack[:n_frames_to_test, :, :]
    n_frames, im_h, im_w = full_img_stack.shape

    prev_image, curr_image = None, full_img_stack[0]
    min_shift, max_shift, step_shift = -5, 5, 1  # must be square (i.e. -nXn, with m step size)
    no_displacement_index = (int((max_shift + abs(min_shift)) / 2), int((max_shift + abs(min_shift)) / 2))
    mx, my = np.meshgrid(np.arange(min_shift, max_shift, step_shift),
                         np.arange(min_shift, max_shift, step_shift))

    nx = mx.ravel()
    ny = my.ravel()
    out_score_per_frame = np.zeros(tuple([n_frames] + list(nx.shape)), dtype=np.float32)

    for img_idx in tqdm(range(len(full_img_stack) - 1)):
        if prev_image is None:
            prev_image = curr_image.copy()
            continue
        # Get curr and prev images
        prev_image, curr_image = curr_image.copy(), full_img_stack[img_idx].copy()
        assert not np.array_equal(prev_image, curr_image)

        # check current frame in relative to the previous one, and fix it according to the min error
        min_shift_index = 0
        fixed_frame2 = curr_image
        curr_min_error = sys.maxsize
        for shift_idx in range(len(nx)):
            shifted_current = np.roll(np.roll(curr_image, nx[shift_idx], axis=1), ny[shift_idx], axis=0)
            curr_shift_error = np.mean(np.square(prev_image - shifted_current))
            if curr_shift_error < curr_min_error:
                curr_min_error = curr_shift_error
                fixed_frame2 = shifted_current.copy()
                min_shift_index = shift_idx
            out_score_per_frame[img_idx][shift_idx] = curr_shift_error

        min_shift_error_index = np.argmin(out_score_per_frame[img_idx])
        shift_x = nx[min_shift_error_index]
        shift_y = ny[min_shift_error_index]

        curr_image = fixed_frame2.copy()
        full_img_stack[img_idx] = fixed_frame2.copy()



    # This section is printing the displacement error -
    # notice that in this printing each frame is in relative to the previous already(!) displaced frame
    # use it just to help you in debugging, testing and validating
    fixed_image_stack = np.zeros_like(full_img_stack)

    shifts = []
    for img_idx in range(1, len(out_score_per_frame)):
        # get the involved frame and the displacements scores between them
        prev_image, curr_image, displacement_scores = full_img_stack[img_idx - 1], full_img_stack[img_idx], \
                                                      out_score_per_frame[img_idx]
        # normalize displacement scores
        displacement_scores = np.divide(displacement_scores, displacement_scores.max(),
                                        out=np.zeros_like(displacement_scores), where=displacement_scores != 0)
        displacement_scores_reshaped = displacement_scores.reshape(mx.shape)
        min_shift_error_index = np.unravel_index(np.argmin(displacement_scores_reshaped),
                                                 displacement_scores_reshaped.shape)
        # flag for visualization only
        displaced = False
        if min_shift_error_index[0] != no_displacement_index[0] or min_shift_error_index[1] != no_displacement_index[1]:
            displaced = True
            movement_in_x, movement_in_y = mx[0, min_shift_error_index[0]], my[min_shift_error_index[1], 0]
            shifts.append((-1 * movement_in_x, -1 * movement_in_y))
        else:
            shifts.append((0, 0))
            # print(
            #     f"displacement of x={movement_in_x}, y={movement_in_y} found between frame #{img_idx - 1} and frame #{img_idx}")
    return shifts


if __name__ == '__main__':
    print(get_shifts(
        r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\211212_CD7_ERK_P38\Erki\211212erki-p38-stiching_s3_tdTom_ORG.tif"), 259)
