from __future__ import print_function
import time
import numpy as np
import pandas as pd
# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from Scripts.DataPreprocessing.load_tracks_xml import *

# For reproducability of the results
from Scripts.PCABuilder import build_pca

np.random.seed(42)
title = "velocity over time- exp {}"
xml_path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Single cell\Tracks_xml\Experiment1_w1Widefield550_s{}_all.xml"

appended_data = []

for i in (1, 2):  # , 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    path = xml_path.format(i)
    full_title = title.format(i)
    _tracks, _ = load_tracks_xml(path)
    tracks = remove_short_tracks(tracks=_tracks, threshold=5)
    for index, x in enumerate(tracks):
        # tracks[index] = x.to_numpy()
        x= x[['a', 'b']]
        feat_cols = ['pixel' + str(i) for i in range(x.shape[1])]
        df = pd.DataFrame(x.to_numpy(), columns=feat_cols)
        appended_data.append(df)
all_tracks_df = pd.concat(appended_data)
rndperm = np.random.permutation(all_tracks_df.shape[0])

plt.gray()
fig = plt.figure(figsize=(16, 7))
for i in range(0, 15):
    ax = fig.add_subplot(3, 5, i + 1, title="Digit: {}".format(str(all_tracks_df.loc[rndperm[i], 'label'])))
    ax.matshow(all_tracks_df.loc[rndperm[i]].values.reshape((28, 28)).astype(float))
plt.show()

build_pca(num_of_components=3, df=all_tracks_df)
