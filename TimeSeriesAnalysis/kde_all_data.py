import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

vid1 = 1
vid2 = 5
dir = "21-12-2021-manual_mastodon_motility-True_intensity-False/140,170 frames ERK, [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]] frames con"

path_con = dir + f"/rev_all_data, motility=True, intensity=False, video #{vid1}"
path_diff = dir +f"/rev_all_data, motility=True, intensity=False, video #{vid2}"
data_con = pickle.load(open(path_con,
    'rb'))
data_diff = pickle.load(open(path_diff,
    'rb'))

data_con["t"] = data_con.index * 30 * 5 / 60
data_diff["t"] = data_diff.index * 30 * 5 / 60
data_diff = data_diff[:len(data_con)]

data_con["target"] = False
data_diff["target"] = True


def plot_(data1, data2, x, y, vid1, vid2):

    fig = plt.figure(figsize=(8, 8))
    sns.kdeplot(
        # ax=ax,
        data=data1,
        x=x,
        y=y,
        hue=data1["t"],
        thresh=.1,
        palette=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    )

    sns.kdeplot(
        # ax=ax,
        data=data2,
        x=x,
        y=y,
        hue=data2["t"],
        thresh=.1,
        palette="YlOrBr"
    )


    plt.show()
    plt.savefig(dir+f"/kde {y}, videos ({vid1},{vid2}).png")


sns.set_theme(style="darkgrid")
# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

cols = ["avg_total", "linearity", "net_distance", "total_distance", "net_total_distance", "monotonicity"]

for col in cols:
    plot_(data_con, data_diff, "confidence", col, vid1, vid2)
