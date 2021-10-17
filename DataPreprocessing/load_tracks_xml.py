import pytrackmate as tm

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 18:34:11 2020

@author: Oron
"""


def load_tracks_xml(location):
    df = tm.trackmate_peak_import(location, get_tracks=True)  ## modified!
    tracks = list()
    ##### Grid count
    for label, labeld_df in df.groupby('label'):
        tracks.append(labeld_df)
    return tracks, df


def remove_short_tracks(tracks, threshold):
    # remove tracks shorter than threshold
    t = 0
    while t < len(tracks):
        if len(tracks[t]) < threshold:
            tracks.pop(t)
        else:
            t += 1
    return tracks


if __name__ == '__main__':

    w=20
    wt_c = [wt * 300 for wt in range(0, 350, w)]

    # path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\New Folder\ERKi_treatment\S8_Nuclei_2015.xml"
    # path = r"C:\Users\Amit\Desktop\Experiment1_w1Widefield550_s2_all_2015.xml"
    # path = r"C:\Users\Amit\Downloads\yon29hazan.xml"
    path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Videos\New Folder\DMSO_control/S3_nuclei_2017.xml"
    track_list, tracks_df = load_tracks_xml(path)
    print("ccccc")
    print(tracks_df.head())
    print(len(tracks_df))
