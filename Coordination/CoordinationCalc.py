# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:30:35 2020
Calculating cos of theta- the angle between the velocity vectors of each cell and it's neighbors.
Constructing a DataFrame called "coordination_outputs" for a later use.
@author: Oron (Amit)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataPreprocessing.load_tracks_xml import load_tracks_xml
import pickle


class CoordinationCalc():

    def __init__(self, SMOOTHING_VAR, NEIGHBORING_DISTANCE, xml_path=None):
        # Params
        self.SMOOTHING_VAR = SMOOTHING_VAR  # A smoothing variable for calculating a cell's velocity vector
        self.NEIGHBORING_DISTANCE = NEIGHBORING_DISTANCE
        # Initiate the output DataFrame
        self.coherence = pd.DataFrame(columns=['Track #', 't0', 'cos_theta'])
        if xml_path is not None:
            self.xml_path = xml_path

    def calc_angle(self, x, y):
        try:
            # The polyfit function returns the polynomial coefficients.
            # pol[0] is the slope of the linear line.
            pol = np.polyfit(x, y, 1, full=False)
            my_angle = np.arctan(pol[0])
        except:
            my_angle = np.pi / 2
        return my_angle

    def get_rand_angle(self):
        N = 1  # number of random angles to generate
        # compute normally distributed values with zero mean and standard deviation of 2 * pi
        thetaNormal = 2 * np.pi * np.random.normal(0, 1, N)
        theta = np.mod(thetaNormal, 2 * np.pi)
        return theta

    def get_neighbors(self, df, x, y, cur_time, validation, rings):
        if validation == True:
            # Get all neighbors
            cur_time_cells = df[df['t_stamp'] == cur_time]
            neighbors = cur_time_cells.sample(frac=.10)

        else:
            if rings == True:
                # Get all neighbors - different distances
                neighbors = df[(np.sqrt((df['x'] - x[0]) ** 2 + (df['y'] - y[0]) ** 2) <= self.NEIGHBORING_DISTANCE) &
                               (df['t_stamp'] == cur_time) &
                               (self.NEIGHBORING_DISTANCE - 70 < np.sqrt(
                                   (df['x'] - x[0]) ** 2 + (df['y'] - y[0]) ** 2))]
            else:
                # Get all neighbors - regular calculation
                neighbors = df[(np.sqrt((df['x'] - x[0]) ** 2 + (df['y'] - y[0]) ** 2) <= self.NEIGHBORING_DISTANCE) &
                               (df['t_stamp'] == cur_time) &
                               (0 < np.sqrt((df['x'] - x[0]) ** 2 + (df['y'] - y[0]) ** 2))]

        # Find unique tracks in the relevant radius
        neighbors = neighbors['label'].unique()

        return neighbors

    def calc_cos_of_angles(self, start_time, end_time, track, tracks01, df, validation, rings):
        # cos_of_angles = np.zeros(track.shape[0] - self.SMOOTHING_VAR + 1)
        cos_of_angles = []
        # loop over all points in tracks (-SMOOTHING_VAR)
        for i, cur_time in enumerate(range(start_time, end_time - self.SMOOTHING_VAR - 1)):
            # Takes SMOOTHING_VAR points at once to smooth the curve
            x = np.asarray(track[(i <= track.t_stamp) & track.t_stamp < i + self.SMOOTHING_VAR]['x'])
            y = np.asarray(track[(i <= track.t_stamp) & track.t_stamp < i + self.SMOOTHING_VAR]['y'])
            my_angle = self.calc_angle(x, y)
            # my_angle = self.get_rand_angle()
            neighbors = self.get_neighbors(df, x, y, cur_time, validation, rings)
            if neighbors.shape[0] != 0:
                angles = np.zeros(neighbors.shape[0])
            else:
                cos_of_angles.append(float('nan'))
                continue

            # Iterate over all adjacent tracks
            for k, num_adj_track in enumerate(neighbors):
                cur_adj_track = tracks01[int(num_adj_track)]
                cur_adj_track_on_time = cur_adj_track[cur_adj_track['t_stamp'] >= cur_time]
                if cur_adj_track_on_time.shape[0] < self.SMOOTHING_VAR:
                    angles[k] = float('nan')
                    break
                # Take SMOOTHING_VAR points at once to smooth the curve
                x_adj = np.asarray(cur_adj_track_on_time.iloc[:self.SMOOTHING_VAR]['x'])
                y_adj = np.asarray(cur_adj_track_on_time.iloc[:self.SMOOTHING_VAR]['y'])
                angles[k] = self.calc_angle(x_adj, y_adj)
                # angles[k] = self.get_rand_angle()

            angles = angles[~np.isnan(angles)]
            try:
                # Calculates the cosine of difference between the cell's angle and its neighbors.
                # Then, calculate the difference's cos, take the absolute value of it.
                # Take the mean of the neighbors values - for smoothing the curve.
                cos_of_angles.append(np.mean(np.abs(np.cos(angles - my_angle))))
            except:
                continue
        return np.asarray(cos_of_angles)

    def build_coordination_df(self, validation, rings=False):
        # Load tha tracks XML (TrackMate's output)
        tracks01, df = load_tracks_xml(self.xml_path)

        # Iterate over all tracks and calculate their velocity vectors, and angles
        for ind, track in enumerate(tracks01, start=0):
            # In case the cell's path is relatively small, ignore it
            if track.shape[0] < 7:
                continue
            print('Track #{}/{}'.format(str(ind), len(tracks01)))
            t_stamp_array = np.asarray(track['t_stamp'])
            start_time = int(np.min(t_stamp_array))
            end_time = int(np.max(t_stamp_array))
            cos_of_angles = self.calc_cos_of_angles(start_time, end_time, track, tracks01, df, validation, rings)
            tmp_df = pd.DataFrame({'Track #': [ind], 't0': [start_time], 'cos_theta': [cos_of_angles]})
            self.coherence = pd.concat([self.coherence, tmp_df])

    def build_coordination_df_(self, tracks01, df, validation=False, rings=False):
        '''for grid calculations'''
        # Iterate over all tracks and calculate their velocity vectors, and angles
        for ind, track in enumerate(tracks01, start=0):
            # In case the cell's path is relatively small, ignore it
            if track is None:
                continue
            if track.shape[0] < 7:
                print(f"shape: {track.shape[0]}")
                continue
            print('Track #{}/{}'.format(str(ind), len(tracks01)))
            t_stamp_array = np.asarray(track['t_stamp'])
            start_time = int(np.min(t_stamp_array))
            end_time = int(np.max(t_stamp_array))
            cos_of_angles = self.calc_cos_of_angles(start_time, end_time, track, tracks01, df, validation, rings)
            tmp_df = pd.DataFrame({'Track #': [ind], 't0': [start_time], 'cos_theta': [cos_of_angles]})
            self.coherence = pd.concat([self.coherence, tmp_df])

    def save_coordinationDF(self, path):
        pickle.dump(self.coherence, open(path, 'wb'))

    def get_coefficients(self, coord_df):
        K = coord_df.shape[0]
        ar = np.zeros((K, 927))
        ar[:] = np.nan
        for i in range(K):
            t0 = int(coord_df['t0'].iloc[i])
            cost = coord_df['cos_theta'].iloc[i]
            N = len(cost)
            ar[i, t0:t0 + N] = cost
        meanar03 = np.nanmean(ar, axis=0)
        coefficient = np.polyfit(np.arange(921), meanar03[:-6], deg=1)
        return coefficient[0]


if __name__ == '__main__':
    print("coordination Calculator")
