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

# from DataPreprocessing.load_tracks_xml import load_tracks_xml
import pickle

np.seterr(divide='ignore', invalid='ignore')
import warnings

warnings.simplefilter('ignore', np.RankWarning)


class CoordinationCalc():

    def __init__(self, SMOOTHING_VAR, NEIGHBORING_DISTANCE, csv_path=None):
        # Params
        self.SMOOTHING_VAR = SMOOTHING_VAR  # A smoothing variable for calculating a cell's velocity vector
        self.NEIGHBORING_DISTANCE = NEIGHBORING_DISTANCE
        # Initiate the output DataFrame
        self.coherence = pd.DataFrame(columns=['Track #', 't0', 'cos_theta'])
        if csv_path is not None:
            self.csv_path = csv_path

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
            cur_time_cells = df[df['Spot frame'] == cur_time]
            neighbors = cur_time_cells.sample(frac=.10)

        else:
            if rings == True:
                # Get all neighbors - different distances
                neighbors = df[(np.sqrt((df['Spot position X (µm)'] - x[0]) ** 2 + (df['Spot position Y (µm)'] - y[0]) ** 2) <= self.NEIGHBORING_DISTANCE) &
                               (df['Spot frame'] == cur_time) &
                               (self.NEIGHBORING_DISTANCE - 70 < np.sqrt(
                                   (df['Spot position X (µm)'] - x[0]) ** 2 + (df['Spot position Y (µm)'] - y[0]) ** 2))]
            else:
                # Get all neighbors - regular calculation
                neighbors = df[(np.sqrt((df['Spot position X (µm)'] - x[0]) ** 2 + (df['Spot position Y (µm)'] - y[0]) ** 2) <= self.NEIGHBORING_DISTANCE) &
                               (df['Spot frame'] == cur_time) &
                               (0 < np.sqrt((df['Spot position X (µm)'] - x[0]) ** 2 + (df['Spot position Y (µm)'] - y[0]) ** 2))]

        # Find unique tracks in the relevant radius
        neighbors = neighbors['Spot track ID'].unique()

        return neighbors

    def calc_cos_of_angles(self, start_time, end_time, track, tracks01, df, validation, rings):
        cos_of_angles = []
        # loop over all points in tracks (-SMOOTHING_VAR)
        for i, cur_time in enumerate(range(start_time, end_time - self.SMOOTHING_VAR - 1)):
            # Takes SMOOTHING_VAR points at once to smooth the curve
            x = np.asarray(track[(i <= track['Spot frame']) & track['Spot frame'] < i + self.SMOOTHING_VAR]['Spot position X (µm)'])
            y = np.asarray(track[(i <= track['Spot frame']) & track['Spot frame'] < i + self.SMOOTHING_VAR]['Spot position Y (µm)'])
            my_angle = self.calc_angle(x, y)  # regular
            # my_angle = self.get_rand_angle() # null model
            neighbors = self.get_neighbors(df, x, y, cur_time, validation, rings)
            if neighbors.shape[0] != 0:
                angles = np.zeros(neighbors.shape[0])
            else:
                cos_of_angles.append(float('nan'))
                continue

            # Iterate over all adjacent tracks
            for k, num_adj_track in enumerate(neighbors):
                # cur_adj_track = tracks01[int(num_adj_track)]
                cur_adj_track = df[df["Spot track ID"] == num_adj_track]
                cur_adj_track_on_time = cur_adj_track[cur_adj_track['Spot frame'] >= cur_time]
                if cur_adj_track_on_time.shape[0] < self.SMOOTHING_VAR:
                    angles[k] = float('nan')
                    break

                # Take SMOOTHING_VAR points at once to smooth the curve
                x_adj = np.asarray(cur_adj_track_on_time.iloc[:self.SMOOTHING_VAR]['Spot position X (µm)'])
                y_adj = np.asarray(cur_adj_track_on_time.iloc[:self.SMOOTHING_VAR]['Spot position Y (µm)'])
                angles[k] = self.calc_angle(x_adj, y_adj)  # regular
                # angles[k] = self.get_rand_angle() # null model

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
        # Load the tracks csv (Mastodon's output)
        # tracks01, df = load_tracks_xml(self.csv_path)
        df = pd.read_csv(csv_path, encoding="cp1252")
        tracks01 = list()
        for label, labeled_df in df.groupby('Spot track ID'):
            labeled_df["label"] = label
            tracks01.append(labeled_df)

        # Iterate over all tracks and calculate their velocity vectors, and angles
        for ind, track in enumerate(tracks01, start=0):
            # In case the cell's path is relatively small, ignore it
            if track.shape[0] < 7:
                continue
            print('Track #{}/{}'.format(str(ind), len(tracks01)))
            t_stamp_array = np.asarray(track['Spot frame'])
            start_time = int(np.min(t_stamp_array))
            end_time = int(np.max(t_stamp_array))
            cos_of_angles = self.calc_cos_of_angles(start_time, end_time, track, tracks01, df, validation, rings)
            tmp_df = pd.DataFrame({'Track #': [track.label], 't0': [start_time], 'cos_theta': [cos_of_angles]})
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
            t_stamp_array = np.asarray(track['Spot frame'])
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
    csv_path = fr"../data/mastodon/test/Nuclei_3-vertices.csv"
    coord = CoordinationCalc(SMOOTHING_VAR=5, NEIGHBORING_DISTANCE=30, csv_path=csv_path)
    coord.build_coordination_df(validation=False)
    coord.save_coordinationDF(r"coordination_outputs/coordination_dfs/manual_tracking/coord_mastodon_s3.pkl")
