import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data_video_3 = pickle.load(open(
    "tmp_motility-True_intensity-False/140,170 frames ERK, [[0, 30], [140, 170], [180, 210]] frames con/all_data , video 3",
    'rb'))
data_video_8 = pickle.load(open(
    "tmp_motility-True_intensity-False/140,170 frames ERK, [[0, 30], [140, 170], [180, 210]] frames con/all_data , video 8",
    'rb'))

data_video_3.index = data_video_3.index * 30 * 5 / 60
data_video_8.index = data_video_8.index * 30 * 5 / 60

data_video_3["target"] = False
data_video_8["target"] = True

df = pd.concat([data_video_3, data_video_8], axis=0)

sns.set_theme(style="darkgrid")
# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
# Draw a contour plot to represent each bivariate density
sns.kdeplot(
    data=data_video_3,
    x="confidence",
    y="net_total_distance",
    hue=data_video_3.index,
    thresh=.1,
    # fill=True,
    palette="muted"
)
sns.kdeplot(
    data=data_video_8,
    x="confidence",
    y="net_total_distance",
    hue=data_video_8.index,
    thresh=.1,
    # fill=True,
)
plt.show()
