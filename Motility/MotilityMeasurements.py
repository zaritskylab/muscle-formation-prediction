import time

from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from Scripts.DataPreprocessing.load_tracks_xml import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
from Scripts.PCABuilder import build_pca, plot_pca


# region measurements
def get_linearity(track):
    x = np.array(track["x"]).reshape((-1, 1))
    y = np.array(track["y"])
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    return r_sq


def get_direction(track):
    right = 0
    left = 0
    zero = 0
    delta = 20
    magnitudes = []
    # print(len(track))
    if len(track) <= delta:
        return 0, 0, 0, 0

    for i in range(0, len(track) - delta - 1, delta):
        x = np.array(track["x"][i:i + delta]).reshape((-1, 1))
        y = np.array(track["y"][i:i + delta])
        model = LinearRegression().fit(x, y)

        x_A = np.array(track["x"])[i]
        x_B = np.array(track["x"])[i + delta]
        y_A = model.intercept_ + model.coef_ * x_A
        y_B = model.intercept_ + model.coef_ * x_B

        x_P = np.array(track["x"])[i]
        y_P = np.array(track["y"])[i]

        x_P_delta = np.array(track["x"])[i + delta + 1]
        y_P_delta = np.array(track["y"])[i + delta + 1]

        # Subtracting co-ordinates of
        # point A from B and P, to
        # make A as origin
        x_B -= x_A
        y_B -= y_A
        x_P_delta -= x_A
        y_P_delta -= y_A

        # Determining cross Product
        cross_product = x_B * y_P_delta - y_B * x_P_delta
        # Return RIGHT if cross product is positive
        if (cross_product > 0):
            right = right + 1
        # Return LEFT if cross product is negative
        elif (cross_product < 0):
            left += 1
        # Return ZERO if cross product is zero
        elif cross_product == 0:
            zero += 1
        # calculate magnitude:
        magnitudes.append(getAngle(a=[x_B, y_B], b=[x_P, y_P], c=[x_P_delta, y_P_delta]))
        # print("left: {}, right: {}, zero: {}".format(left, right, zero))
    p_turn = right / (left + right + zero + 0.000001)
    if len(magnitudes) > 0:
        min_theta = np.min(magnitudes)
        max_theta = np.max(magnitudes)
        mean_theta = np.mean(magnitudes)
    else:
        min_theta, max_theta, mean_theta = 0, 0, 0
    return p_turn, min_theta, max_theta, mean_theta


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


# print(getAngle((5, 0), (0, 0), (0, 5)))


def get_monotonicity(track):
    x = np.array(track["x"]).reshape((-1, 1))
    y = np.array(track["y"])
    rho, p_value = stats.spearmanr(x, y)
    return rho


def velocisty_over_time(xml_path):
    tracks, _ = load_tracks_xml(xml_path)
    tracks = remove_short_tracks(tracks=tracks, threshold=5)

    for t in range(len(tracks)):
        track_df = tracks[t]
        df_size = len(track_df) - 1
        velocities = np.ndarray(shape=(df_size, 2))
        xs = track_df["x"].tolist()
        ys = track_df["y"].tolist()
        for i in range(df_size - 1):
            velocities[i][0] = (xs[i + 1] - xs[i]) * (10 ^ 4) / 58.4564 * 40  # um/h
            velocities[i][1] = (ys[i + 1] - ys[i]) * (10 ^ 4) / 58.4564 * 40  # um/h
        dataSet = pd.DataFrame({"x_velocities": velocities[:, 0], "y_velocities": velocities[:, 1],
                                "velocities": np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)})

        top_row = []
        top_row.insert(0, {"x_velocities": dataSet["x_velocities"][0],
                           "y_velocities": dataSet["y_velocities"][0],
                           "velocities": dataSet["velocities"][0]})
        dataSet = pd.concat([pd.DataFrame(top_row), dataSet], ignore_index=True)

        tracks[t]["x_velocities"] = dataSet["x_velocities"].values
        tracks[t]["y_velocities"] = dataSet["y_velocities"].values
        tracks[t]["velocities"] = dataSet["velocities"].values

    return tracks


def plot_velocity_over_time(velocity_tracks, title):
    mean_velocity = []
    max_velocity = []
    min_velocity = []
    median_velocity = []

    for t in range(927):
        v_t = []
        for track in velocity_tracks:
            if True in (track["t_stamp"] == t).unique():
                v_t.append(track[track["t_stamp"] == t].velocities.values[0])
        mean_velocity.append(np.mean(v_t))
        min_velocity.append(np.min(v_t))
        max_velocity.append(np.max(v_t))
        median_velocity.append(np.median(v_t))

    time = range(927)
    plt.plot(time, min_velocity, '.')
    plt.plot(time, max_velocity, '.')
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.5, 0.05))
    plt.plot(time, mean_velocity, '.')
    plt.plot(time, median_velocity, '.')
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.title(title)
    plt.legend(['min', 'max', 'mean', 'median '])
    plt.savefig("velocity over time/" + str(title))
    plt.show()


def cell_in_motion(track):
    track_len = len(track)
    motion_threshold = 0.05
    motion_count = track[track["velocities"] > motion_threshold].count()["velocities"]
    motion_proportion = motion_count / track_len
    motion_df = track[track["velocities"] < motion_threshold]
    return motion_proportion, motion_df


def avg_moving_speed(track):
    motion_threshold = 0.05
    im_motion_df = track[track["velocities"] > motion_threshold]
    avg_moving_speed = np.mean(im_motion_df["velocities"])
    return avg_moving_speed


def plot_exps_mean_no_motion_proportion(all_props):
    indexes_con = [0, 1, 6, 7, 8, 9]
    indexes_1 = [i + 1 for i in indexes_con]
    plt.bar(indexes_1, np.array(all_props)[indexes_con])
    indexes_dif = [2, 3, 4, 5, 10, 11]
    indexes_1 = [i + 1 for i in indexes_dif]
    plt.bar(indexes_1, np.array(all_props)[indexes_dif])
    plt.xlabel("exp num")
    plt.xticks(range(1, 13))
    plt.yticks(np.arange(0, 0.6, 0.1))
    plt.ylabel("mean proportion")
    plt.title("mean non motion proportion per experiment")
    plt.legend(["control", "diff"])
    plt.show()


def plot_no_prop_motion(props):
    plt.plot(props, '.')
    plt.xlabel("track number")
    plt.ylabel("time in no motion/total time")
    plt.title("no motion proportion- exp {}".format(i))
    plt.show()


def plot_no_motion_location(no_motion_df):
    plt.scatter(no_motion_df["x"], no_motion_df["y"], c=no_motion_df["t"])
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("coordinates of low motion exp - {}".format(i))
    plt.show()


def get_motion_proportions(velocity_tracks):
    props = []
    appended_data = []
    for track in velocity_tracks:
        motion_prop, cell_in_motion_df = cell_in_motion(track)
        appended_data.append(cell_in_motion_df)
        props.append(motion_prop)
    motion_df = pd.concat(appended_data)
    return motion_df, props


# Calculate distance between 2 points
def get_distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x2 - x1, 2) +
                     math.pow(y2 - y1, 2) * 1.0)


def get_total_distance(track):
    xs = track["x"].tolist()
    ys = track["y"].tolist()
    total_distance = 0
    for i in range(len(track) - 1):
        total_distance += get_distance(xs[i], ys[i], xs[i + 1], ys[i + 1])
    return total_distance


def calc_distances(xml_path):
    tracks, _ = load_tracks_xml(xml_path)
    tracks = remove_short_tracks(tracks=tracks, threshold=5)
    lengths = []
    net_distances = []
    total_distances = []
    for track in tracks:
        net_distance = get_distance(x1=track[track["t"] == int(np.min(track["t"]))]["x"].values[0],
                                    y1=track[track["t"] == int(np.min(track["t"]))]["y"].values[0],
                                    x2=track[track["t"] == int(np.max(track["t"]))]["x"].values[0],
                                    y2=track[track["t"] == int(np.max(track["t"]))]["y"].values[0])
        total_distance = get_total_distance(track)
        lengths.append(len(track))
        net_distances.append(net_distance)
        total_distances.append(total_distance)
        # plt.scatter(track["x"], track["y"], c=track["t"])
        # plt.show()
    props = np.array(net_distances) / np.array(total_distances)
    return lengths, net_distances, total_distances, props


def get_net_total_proportion(net_distance, total_distance):
    if total_distance == 0:
        distance_prop = 0
    else:
        distance_prop = net_distance / total_distance
    return distance_prop


def get_msd(track):
    time_lag = 40
    msds = []
    if len(track) <= time_lag:
        return 0

    for i in range(0, time_lag - 1):
        x_t = np.array(track["x"])[i:i + time_lag]
        x_tao = np.array(track["x"])[time_lag]
        msds = x_tao - x_t
    msd = np.mean(msds)
    # print("msd", msd)
    # print("np.log2(msd)", np.log2(msd+1))
    # print("np.log2(time_lag)", np.log2(time_lag))
    alpha = np.log2(msd + 1) / np.log2(time_lag)
    return alpha


# endregion

def get_measurements(track, lst):
    net_distance = get_distance(x1=track[track["t"] == int(np.min(track["t"]))]["x"].values[0],
                                y1=track[track["t"] == int(np.min(track["t"]))]["y"].values[0],
                                x2=track[track["t"] == int(np.max(track["t"]))]["x"].values[0],
                                y2=track[track["t"] == int(np.max(track["t"]))]["y"].values[0])
    total_distance = get_total_distance(track)
    net_total_distance = get_net_total_proportion(net_distance, total_distance)
    min_velocity = np.min(track["velocities"])
    max_velocity = np.max(track["velocities"])
    mean_velocity = np.mean(track["velocities"])
    cell_notion_prop, _ = cell_in_motion(track)
    avg_moving_speed_val = avg_moving_speed(track)
    linearity = get_linearity(track)
    monotonicity = get_monotonicity(track)
    p_turn, min_theta, max_theta, mean_theta = get_direction(track)
    msd_alpha = get_msd(track)

    if i in (1, 2, 7, 8, 9, 10):
        differentiation = 0
    elif i in (3, 4, 5, 6, 11, 12):
        differentiation = 1
    lst.append([net_distance, total_distance, net_total_distance,
                min_velocity, max_velocity, mean_velocity,
                cell_notion_prop, avg_moving_speed_val, differentiation, linearity, monotonicity,
                p_turn, min_theta, max_theta, mean_theta, msd_alpha])


def normalize_data(df):
    result = df.copy()
    result.replace([np.inf], np.nan, inplace=True)
    result.loc[result["max_velocity"] > 10000, "max_velocity"] = np.nan
    result.loc[result["mean_velocity"] > 1000, "mean_velocity"] = np.nan
    result.loc[result["Average moving speed"] > 1000, "mean_velocity"] = np.nan
    result = result.fillna(result.mean())

    for feature_name in result.columns.drop("differentiation status"):
        max_value = result[feature_name].max()
        min_value = result[feature_name].min()
        result[feature_name] = (result[feature_name] - min_value) / (max_value - min_value)
    return result.copy()


title = "velocity over time- exp {}"
xml_path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Single cell\Tracks_xml\Experiment1_w1Widefield550_s{}_all.xml"

all_props = []
mean_lengths = []
mean_net_distances = []
mean_total_distances = []
mean_props = []
columns = ['total_distance', 'net_distance', 'progressivity',
           'min_velocity', 'max_velocity', 'mean_velocity',
           'Time Spent Moving', 'Average moving speed', 'differentiation status', 'linearity',
           'monotonicity', 'Proportion of right turns', 'Minimum turn magnitude',
           'Maximum turn magnitude', 'Average turn magnitude', 'Mean squared displacement']

# all_data_df = pd.DataFrame(columns=columns)
# part1_df = pd.DataFrame(columns=columns)
lst_all_data = []
lst_part1 = []
lst_part2 = []
for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):  # , 4, 5, 6, 7, 8, 9, 10, 11, 12
    path = xml_path.format(i)
    full_title = title.format(i)
    velocity_tracks = velocisty_over_time(xml_path=path)

    for track in velocity_tracks:

        get_measurements(track, lst_all_data)

        part1_track = track.drop(track[track["t"] > 600].index)
        if len(part1_track) > 0:
            get_measurements(part1_track, lst_part1)

        part2_track = track.drop(track[track["t"] < 600].index)
        if len(part2_track) > 0:            get_measurements(part2_track, lst_part2)

all_data_df = pd.DataFrame(lst_all_data, columns=columns)
part1_df = pd.DataFrame(lst_part1, columns=columns)
part2_df = pd.DataFrame(lst_part2, columns=columns)

all_data_df = normalize_data(all_data_df)
part1_df = normalize_data(part1_df)
part2_df = normalize_data(part2_df)

all_data_df.hist()


def make_plot_pca(df, title):
    principal_df_diff, pca = build_pca(num_of_components=3, df=df)
    print('{}: Explained variation per principal component: {}'.format(title, pca.explained_variance_ratio_))
    plot_pca(principal_df=principal_df_diff, title=title)


# PCA
diff_df = all_data_df[all_data_df["differentiation status"] == 1]
control_df = all_data_df[all_data_df["differentiation status"] == 0]
make_plot_pca(diff_df, title="differentiation")
make_plot_pca(control_df, title="control")
all_data_df.drop('differentiation status', axis='columns', inplace=True)
make_plot_pca(all_data_df, title="all data")

diff_part1 = part1_df[part1_df["differentiation status"] == 1]
control_part1 = part1_df[part1_df["differentiation status"] == 0]
make_plot_pca(diff_part1, title="differentiation p1")
make_plot_pca(control_part1, title="control p1")
part1_df.drop('differentiation status', axis='columns', inplace=True)
make_plot_pca(part1_df, title="all data p1")

diff_part2 = part2_df[part2_df["differentiation status"] == 1]
control_part2 = part2_df[part2_df["differentiation status"] == 0]
make_plot_pca(diff_part2, title="differentiation p2")
make_plot_pca(control_part2, title="control p2")
part2_df.drop('differentiation status', axis='columns', inplace=True)
make_plot_pca(part2_df, title="all data p2")

# rndperm = np.random.permutation(all_data_df.shape[0])

# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="principal component 1", y="principal component 2",
#     hue="principal component 3",
#     palette=sns.color_palette("hls", len(principal_df['principal component 3'].unique())),
#     data=principal_df.loc[rndperm, :],
#     legend=False,
#     # alpha=0.3
# )
# plt.show()

# # TSNE
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(analysed_df)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
# df_subset = analysed_df.loc[rndperm,:].copy()
# df_subset['tsne-2d-one'] = tsne_results[:,0]
# df_subset['tsne-2d-two'] = tsne_results[:,1]
#
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="tsne-2d-two",
#     palette=sns.color_palette("hls", len(df_subset['tsne-2d-two'].unique())),
#     data=df_subset.loc[rndperm, :],
#     legend="full",
#     alpha=0.3
# )

# plot_velocity_over_time(velocity_tracks, full_title)
# no_motion_df, props = get_no_motion_proportions(velocity_tracks)
# all_props.append(np.mean(props))
# plot_no_prop_motion(props)
# plot_no_motion_location(no_motion_df)
# plot_exps_mean_no_motion_proportion(all_props)

# lengths, net_distances, total_distances, props = calc_distances(xml_path=path)
# mean_lengths.append(np.mean(lengths))
# mean_net_distances.append(np.mean(net_distances))
# mean_total_distances.append(np.mean(total_distances))
# mean_props.append(np.mean(props))
#
# indexes_con = [0, 1, 6, 7, 8, 9]
# indexes_dif = [2, 3, 4, 5, 10, 11]
# plt.bar(np.arange(1, 7, 1) - 0.2, np.array(props)[indexes_con], width=0.2, align='edge')
# plt.bar(np.arange(1, 7, 1), np.array(props)[indexes_dif], width=0.2, align='edge')
# plt.xlabel("exp num")
# # plt.xticks(range(1, 13))
# plt.yticks(np.arange(0, 8, 0.5))
# plt.ylabel("mean net:total proportion")
# plt.title("mean proportion net/total per experiment")
# plt.legend(["control", "diff"])
# plt.show()
