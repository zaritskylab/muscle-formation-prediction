import joblib
from DataPreprocessing.load_tracks_xml import *
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
import matplotlib.pyplot as plt
from skimage import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
# from Motility.MotilityMeasurements import get_monotonicity, get_direction
# from PCABuilder import build_pca
# from ts_fresh import get_x_y, get_single_cells_diff_score_plot
from sklearn.preprocessing import LabelEncoder
from os import path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import shap
from random import sample
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")


def extract_features_for_prediction(X, features_df=None):
    if features_df is None:
        features_df = pickle.load(open("TimeSeriesAnalysis/train_features_500_750", 'rb'))

    X_extracted = extract_features(X, column_id="label", column_sort="t")
    impute(X_extracted)
    X_extracted = X_extracted[features_df.columns]
    return X_extracted


def plot_pca(principal_df, path):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue='target',
        palette=sns.color_palette("hls", len(principal_df['target'].unique())),
        data=principal_df,
        legend='brief',
        alpha=0.3
    )
    plt.title("exp 5")
    plt.savefig(path+ "/pca.png")
    plt.show()


def plot_featurs_boxplot(X_extracted, f_list):
    for ind, col in enumerate(f_list):
        ax = sns.boxplot(x="target", y=str(col), hue="target",
                         data=X_extracted, palette="Set3")
        plt.title(col)
        plt.show()


def pca(X_extracted, y,):
    # build PCA on all data
    principal_df, pca = build_pca(2, X_extracted)
    principal_df["target"] = y
    # initializing an object of class LabelEncoder
    labelencoder = LabelEncoder()
    # fitting and transforming the desired categorical column.
    principal_df["target"] = labelencoder.fit_transform(principal_df["target"])
    # plot_pca(principal_df)
    return principal_df


def get_patches(track, bf_video):
    image_size = 32
    im = io.imread(bf_video)
    crops = []
    for i in range(len(track)):
        x = int(track.iloc[i]["x"])
        y = int(track.iloc[i]["y"])
        single_cell_crop = im[int(track.iloc[i]["t_stamp"]), y - image_size:y + image_size,
                           x - image_size:x + image_size]
        crops.append(single_cell_crop)
        # cv2.imwrite("single_cell_crop.tif", single_cell_crop)
    return crops


def get_shap_explainations(model):
    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    shap.plots.waterfall(shap_values[0])

    # visualize the first prediction's explanation with a force plot
    shap.plots.force(shap_values[0])

    # visualize all the training set predictions
    shap.plots.force(shap_values)

    # create a dependence scatter plot to show the effect of a single feature across the whole dataset
    # shap.plots.scatter(shap_values[:, "RM"], color=shap_values)

    # summarize the effects of all the features
    shap.plots.beeswarm(shap_values)

    shap.plots.bar(shap_values)

    shap.summary_plot(shap_values, X)
    #
    # shap.dependence_plot(
    #     ("Age", "Sex"),
    #     shap_interaction_values, X.iloc[:2000,:],
    #     display_features=X_display.iloc[:2000,:]
    # )


def get_prob_over_track(clf, track, window, features_df):
    true_prob = []
    for i in range(0, len(track), window):
        track_portion = track[i:i + window]
        X = extract_features_for_prediction(X=track_portion, features_df=features_df)
        probs = clf.predict_proba(X)
        true_prob.append(probs[0][1])
        print(f"track portion: [{i}:{i + window}]")
        print(clf.classes_)
        print(probs)
    return true_prob


def features_prob(track):
    window = 40
    # calc monotonicity:
    # monotonicity = get_monotonicity(track)
    # p_turn, min_theta, max_theta, mean_theta = get_direction(track)


def plot_cell_probability(cell_ind, bf_video, clf, track, window, target_val, features_df, text, path=None):
    '''
    plot image crops together with their probability of being differentiated
    :param track:
    :param window:
    :param target_val:
    :return:
    '''
    true_prob = get_prob_over_track(clf, track, window, features_df)
    cell_patches = get_patches(track, bf_video)

    X_full_track = extract_features_for_prediction(X=track, features_df=features_df)
    pred = clf.predict(X_full_track)
    total_prob = clf.predict_proba(X_full_track)[0][1]

    windowed_cell_patches = [cell_patches[i] for i in range(0, len(cell_patches), window)]
    track_len = len(track)
    label_time = [(track.iloc[val]['t']) * 90 / 60 / 60 for val in range(0, track_len, 2 * window)]

    plt.figure(figsize=(20, 6))
    fig, ax = plt.subplots()
    ax.scatter(range(0, track_len, window), true_prob)
    for x0, y0, patch in zip(range(0, track_len, window), true_prob, windowed_cell_patches):
        ab = AnnotationBbox(OffsetImage(patch, 0.32), (x0, y0), frameon=False)
        ax.add_artist(ab)
        plt.gray()
        plt.grid()
    ax.plot(range(0, track_len, window), true_prob)
    plt.xticks(range(0, track_len, 2 * window), labels=np.around(label_time, decimals=1))
    plt.title(f"cell #{cell_ind} probability of differetniation")
    plt.text(0.1, 0.9, f'target: {target_val}', ha='center', va='center', transform=ax.transAxes)
    plt.text(0.2, 0.8, f'total track prediction: {pred[0]}, {total_prob}', ha='center', va='center',
             transform=ax.transAxes)
    plt.text(0.5, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.xlabel("time (h)")
    plt.ylabel("prob")
    plt.ylim(0.2, 1, 0.05)
    plt.grid()
    if path:
        print(path)
        plt.savefig(path + ".png")
    plt.show()


if __name__ == '__main__':
    print("Let's go!")
    do_pca = False
    plot_cells = True
    do_shap = False
    motility = False
    intensity = True
    lst_videos = [7, 5]

    lst_pickles = ["clf_2_4", "clf_2_4_intensity_all_tracks", "clf_2_4_intensity_tracks500_750.joblib",
                   "clf_features_200_250_intensity", "clf_features_200_250_intensity_600",
                   "clf_features_200_250_intensity_new", "clf_features_200_250_motility_600",
                   "clf_features_350_400_intensity_550.joblib"]
    lst_features_dfs = ["train_features_2_4_intensity", "train_features_200_250_intensity",
                        "train_features_200_250_intensity_600", "train_features_200_250_intensity_new",
                        "train_features_200_250_motility_600", "train_features_350_400_intensity_550"]

    classifiers_features = zip(lst_pickles, lst_features_dfs)

    pickle_name = "train_features_200_250_motility_600"

    X, y = get_x_y(lst_videos, motility, intensity)

    if path.exists(pickle_name):
        X_extracted = pickle.load(open(pickle_name, 'rb'))
    else:
        X_extracted = extract_features_for_prediction(X)
        pickle.dump(X_extracted, open(pickle_name, 'wb'))

    # load, no need to initialize the clf
    clf = joblib.load("TimeSeriesAnalysis/clf_features_200_250_motility_600.joblib")
    pred = clf.predict(X_extracted)
    prob = clf.predict_proba(X_extracted)

    if plot_cells:
        xml_path = r"data/tracks_xml/pixel_ratio_1/Experiment1_w1Widefield550_s11_all_pixelratio1.xml"
        bf_video = r"data/videos/BrightField_pixel_ratio_1/Experiment1_w2Brightfield_s11_all_pixelratio1.tif"
        tracks, df = load_tracks_xml(xml_path)

        # get_single_cells_diff_score_plot(tracks, clf, X_extracted)

        sampled_tracks = sample(list(enumerate(tracks)), 3500)

        for cell_ind, curr_track in sampled_tracks:
            if len(curr_track) < 600: continue
            print(len(curr_track))
            plot_cell_probability(track=curr_track, window=60, target_val=True, features_df=X_extracted,
                                  text="clf_features_200_250_intensity_new")

    if do_shap:
        get_shap_explainations(clf)

    if do_pca:
        pca(X_extracted, y)

    # # plot each feature's plot
    # X_extracted["prob"] = prob[:, 1]
    # X_extracted["target"] = y
    # # initializing an object of class LabelEncoder
    # labelencoder = LabelEncoder()
    # # fitting and transforming the desired categorical column.
    # X_extracted["target"] = labelencoder.fit_transform(X_extracted["target"])

    # TODO: 1. plot image crops together with their probability of being differentiated- V
    # TODO: 2. shap feature importance
    # TODO: 3. get velocity, direction mof movement, monotonicity - over the probability to differentiate
    # TODO: 4. compare #3 values for several cells with different differentiation probability.
