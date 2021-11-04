from DataPreprocessing.load_tracks_xml import load_tracks_xml, remove_short_tracks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque


def displacement_over_time(xml_path):
    """
    Calculates the change in displacement over time, in X axis, Y axis and total displacement (sqrt(x^2 +y^2))
    :param xml_path: path to the tracks xml file
    :return: tracks with calculated displacement
    """

    def column_displacement(col):
        """
        Calculates the change in displacement over time for a single dataframe column
        :param col: dataframe column's values
        :return: set of displacement value for each frame
        """
        vals = np.array(col)
        vals_copy = vals.copy()
        vals_copy = np.roll(vals_copy, 1)
        velocities = (vals_copy - vals)
        velocities[0] = velocities[1]
        return velocities

    # load tracks list
    tracks, _ = load_tracks_xml(xml_path)

    # remove tracks that are shorter than 10 frames
    tracks = remove_short_tracks(tracks=tracks, threshold=10)

    # calculate displacement for each track
    for t in range(len(tracks)):
        track_df = tracks[t]
        x_velocities = column_displacement(track_df["x"])
        y_velocities = column_displacement(track_df["y"])

        tracks[t]["x_velocities"] = x_velocities
        tracks[t]["y_velocities"] = y_velocities
        tracks[t]["velocities"] = np.sqrt(x_velocities ** 2 + y_velocities ** 2)

    return tracks


def plot_displacement_over_time(velocity_tracks, title, video_frame_num):
    """
    Calculates min, max, median and avg values for all cells in a given dataframe, over the video's time.
    Then, the method plots the displacement values for each time frame
    :param velocity_tracks: list of dataframes for each cell, contains displacement values
    :param title: title of the plot
    :return:
    """
    mean_velocity = []
    max_velocity = []
    min_velocity = []
    median_velocity = []

    # calculate mean, min, max and median values of all cells, for each frame
    for t in range(video_frame_num):
        v_t = []
        for track in velocity_tracks:
            if True in (track["t_stamp"] == t).unique():
                v_t.append(track[track["t_stamp"] == t].velocities.values[0])
        mean_velocity.append(np.mean(v_t))
        min_velocity.append(np.min(v_t))
        max_velocity.append(np.max(v_t))
        median_velocity.append(np.median(v_t))

    # print(mean_velocity)
    # print(max_velocity)
    # print(min_velocity)
    # print(median_velocity)

    time = range(video_frame_num)
    plt.plot(time, min_velocity, '.')
    plt.plot(time, max_velocity, '.')
    # plt.ylim(0, 1)
    # plt.yticks(np.arange(0, 0.5, 0.05))
    plt.plot(time, mean_velocity, '.')
    plt.plot(time, median_velocity, '.')
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.title(title)
    plt.legend(['min', 'max', 'mean', 'median '])
    plt.savefig("velocity over time/" + str(title))
    plt.show()


if __name__ == '__main__':
    video_num = 13
    video_frame_num = 302
    path = fr"../data/tracks_xml/pixel_ratio_1/1804/s{video_num}_all_pixel_ratio_1.xml"

    velocity_tracks = displacement_over_time(xml_path=path)
    plot_displacement_over_time(velocity_tracks, f"velocity measures over time video #{video_num}", video_frame_num)
