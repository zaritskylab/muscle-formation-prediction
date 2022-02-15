import os
import pickle
import seaborn as sns
import joblib
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndi
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute


def get_position(ind, df):
    x = int(df.iloc[ind]["Spot position X (µm)"] / 0.462)
    y = int(df.iloc[ind]["Spot position Y (µm)"] / 0.462)
    spot_frame = int(df.iloc[ind]["Spot frame"])
    return x, y, spot_frame


def get_centered_image(ind, df, im_actin, window_size):
    x, y, spot_frame = get_position(ind, df)
    cropped = im_actin[spot_frame][x - window_size:x + window_size, y - window_size: y + window_size]
    return cropped


def get_intensity_measures_roi(img):
    cv2.imwrite('temp.png', img)
    img = cv2.imread('temp.png', 0)
    new_rgb = np.dstack([img, img, img])
    imgGry = cv2.cvtColor(new_rgb, cv2.COLOR_BGR2GRAY)

    im = ndi.gaussian_filter(imgGry, sigma=l / (4. * n))
    mask = im > im.mean()
    label_im, nb_labels = ndi.label(mask)

    # Find the largest connected component
    sizes = ndi.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)

    # Now that we have only one connected component, extract it's bounding box
    slice_x, slice_y = ndi.find_objects(label_im == 1)[0]
    roi = imgGry[slice_x, slice_y]
    return roi.min(), roi.max(), roi.mean(), roi.sum()


def get_single_cell_intensity_measures(label, df, im_actin, window_size):
    try:
        df_measures = pd.DataFrame(columns=["min", "max", "mean", "sum", "Spot track ID", "Spot frame", "x", "y", ])
        for i in range(len(df)):  # len(df)
            img = get_centered_image(i, df, im_actin, window_size)
            # min_i, max_i, mean_i, sum_i = get_intensity_measures_roi(img)
            min_i, max_i, mean_i, sum_i = img.min(), img.max(), img.mean(), img.sum()
            x, y, spot_frame = get_position(i, df)
            data = {"min": min_i, "max": max_i, "mean": mean_i, "sum": sum_i, "Spot track ID": label,
                    "Spot frame": spot_frame,
                    "x": x, "y": y}
            df_measures = df_measures.append(data, ignore_index=True)
    except:
        return pd.DataFrame()
    return df_measures


def get_intensity_measures_df(csv_path, video_actin_path, window_size):
    df = pd.read_csv(csv_path, encoding="cp1252")
    df = df[df["manual"] == 1]
    im_actin = io.imread(video_actin_path)
    total_df = pd.DataFrame()
    for label, label_df in df.groupby("Spot track ID"):
        df_measures = get_single_cell_intensity_measures(label=label, df=label_df, im_actin=im_actin,
                                                         window_size=window_size)
        if not df_measures.empty:
            total_df = pd.concat([total_df, df_measures], axis=0)
    return total_df


def get_intensity_measures_df_df(df, video_actin_path, window_size):
    im_actin = io.imread(video_actin_path)
    total_df = pd.DataFrame()
    for label, label_df in df.groupby("Spot track ID"):
        df_measures = get_single_cell_intensity_measures(label=label, df=label_df, im_actin=im_actin,
                                                         window_size=window_size)
        if not df_measures.empty:
            total_df = pd.concat([total_df, df_measures], axis=0)
    return total_df


def get_single_measure_vector_df(intentsity_measures_df, measure_name):
    single_measure_df = pd.DataFrame(columns=[i for i in range(262)])

    for lable, lable_df in intentsity_measures_df.groupby("Spot track ID"):
        frame_label_df = lable_df[["Spot frame", measure_name]]
        frame_label_df.index = frame_label_df["Spot frame"].astype(int)
        frame_label_df = frame_label_df[~frame_label_df.index.duplicated()]
        single_measure_df = single_measure_df.append(frame_label_df["mean"])
    single_measure_df.index = [i for i in range(len(single_measure_df))]

    return single_measure_df


def plot_intensity_over_time(measure_name, control_df, erk_df, path):
    def plot_df_intensity(df, color):
        avg_vals = df.mean()
        std_vals = df.std()
        plt.plot(avg_vals, color=color)
        p_std = np.asarray(avg_vals) + np.asarray(std_vals)
        m_std = np.asarray(avg_vals) - np.asarray(std_vals)
        plt.fill_between(range(0, 262), m_std, p_std, alpha=0.5, color=color)

    plot_df_intensity(erk_df, "DarkOrange")
    plot_df_intensity(control_df, "Blue")
    plt.xticks(np.arange(0, 262, 30), labels=np.around(np.arange(0, 262, 30) * 5 / 60, decimals=1))
    plt.xlabel("time(h)")
    plt.ylabel("mean intensity")
    plt.title(f"{measure_name} intensity over time")
    plt.savefig(path)
    plt.show()


def plot_centerd_image(ind, df, label, im_actin, im_nuclei):
    x, y, spot_frame = get_position(ind, df)
    fig = plt.figure(figsize=(10, 10))
    plt.grid(None)
    plt.gray()
    plt.imshow(im_actin[spot_frame], cmap='pink')
    plt.imshow(im_nuclei[spot_frame], alpha=0.15, cmap='cool')
    plt.xlim(x - window_size, x + window_size)
    plt.ylim(y - window_size, y + window_size)
    plt.scatter(x, y, 8000, color="none", edgecolor="yellow")
    plt.axis('off')
    plt.suptitle(f"single cell, label:{label}, frame:{spot_frame}, time:{spot_frame * 5 / 60}", fontsize=20)
    plt.grid(None)


# <editor-fold desc="model">
def evaluate(clf, X_test, y_test):
    predicted = cross_val_predict(clf, X_test, y_test, cv=5)
    report = classification_report(y_test, predicted)
    auc_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(report)
    print(auc_score)
    return report, auc_score


def get_unique_indexes(y):
    idxs = y.index.unique()
    lst = [y[idx] if isinstance(y[idx], np.bool_) else y[idx].iloc[0] for idx in idxs]
    y_new = pd.Series(lst, index=idxs).sort_index()
    return y_new


def drop_columns(df):
    return df[['min', 'max', 'mean', 'sum', 'Spot track ID', 'Spot frame', 'target']]


def normalize(df):
    columns = [e for e in list(df.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
    scaler = StandardScaler()  # create a scaler
    df[columns] = scaler.fit_transform(df[columns])  # transform the feature
    return df


def concat_dfs(diff_df, con_df, diff_t_window=None, con_t_windows=None):
    def set_indexes(df, target, max_val):
        df["Spot track ID"] = df["Spot track ID"] + max_val
        max_val = df["Spot track ID"].max() + 1
        df['target'] = np.array([target for i in range(len(df))])
        return df, max_val

    max_val = 0
    diff_start, diff_end = diff_t_window
    window_size = diff_end - diff_start

    # Erk video
    # Cut the needed time window
    print(diff_df.columns)
    diff_df = diff_df[(diff_df["Spot frame"] >= diff_start) & (diff_df["Spot frame"] < diff_end)]

    # control video
    # Cut the needed time window
    control_df = pd.DataFrame()
    new_label = max(con_df['Spot track ID'].unique()) + 1
    for start, end in con_t_windows:
        tmp_df = con_df[(con_df["Spot frame"] >= start) & (con_df["Spot frame"] < end)]
        for label, label_df in tmp_df.groupby('Spot track ID'):
            if len(label_df) == window_size:
                new_label += 1
                label_df["Spot track ID"] = new_label
                control_df = control_df.append(label_df)
    con_df = control_df.copy()

    diff_df, max_val = set_indexes(diff_df, target=True, max_val=max_val)
    con_df, _ = set_indexes(con_df, target=False, max_val=max_val)
    total_df = pd.concat([diff_df, con_df], ignore_index=True)
    return total_df


def prep_data(diff_df, con_df, diff_t_window, con_t_windows):
    df = concat_dfs(diff_df, con_df, diff_t_window, con_t_windows)
    df = drop_columns(df)
    df = normalize(df)

    df = df.sample(frac=1).reset_index(drop=True)

    y = pd.Series(df['target'])
    y.index = df['Spot track ID']
    y = get_unique_indexes(y)
    df = df.drop("target", axis=1)
    return df, y


def train(X_train, y_train):
    clf = RandomForestClassifier(max_depth=8)
    clf.fit(X_train, y_train, )
    return clf


# </editor-fold>

# <editor-fold desc="plot model metrics">
def plot_roc(clf, X_test, y_test, path):
    # plt.figure(figsize=(20, 6))
    # roc curve for models
    fpr1, tpr1, thresh1 = roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)

    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    plt.style.use('seaborn')
    # plot roc curves
    plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Random Forest')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    # title
    plt.title('ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(path + "/" + 'ROC', dpi=300)
    plt.show()


def build_pca(num_of_components, df):
    '''
    The method creates component principle dataframe, with num_of_components components
    :param num_of_components: number of desired components
    :param df: encoded images
    :return: PCA dataframe
    '''
    pca = PCA(n_components=num_of_components)
    principal_components = pca.fit_transform(df)
    colomns = ['principal component {}'.format(i) for i in range(1, num_of_components + 1)]
    principal_df = pd.DataFrame(data=principal_components, columns=colomns)
    return principal_df, pca


def plot_pca(principal_df, pca, path):
    '''
    The method plots the first 3 dimensions of a given PCA
    :param principal_df: PCA dataframe
    :return: no return value
    '''
    variance = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue='principal component 3',
        palette=sns.color_palette("hls", len(principal_df['principal component 3'].unique())),
        data=principal_df,
        legend=False,
        alpha=0.3
    )
    plt.xlabel(f"PC1 ({variance[0]}) %")
    plt.ylabel(f"PC2 ({variance[1]}) %")
    plt.title("PCA")
    plt.savefig(path + "/pca.png")
    plt.show()


def feature_importance(clf, feature_names, path):
    # Figure Size
    top_n = 30
    fig, ax = plt.subplots(figsize=(16, 9))

    sorted_idx = clf.feature_importances_.argsort()

    ax.barh(feature_names[sorted_idx[-top_n:]], clf.feature_importances_[sorted_idx[-top_n:]])
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=50)

    plt.xlabel("Random Forest Feature Importance")
    plt.title('Feature Importance Plot')
    plt.savefig(path + "/feature importance.png")
    plt.show()


# </editor-fold>


# <editor-fold desc="utils">
def open_dirs(main_dir, inner_dir):
    if not os.path.exists(main_dir):
        os.mkdir(main_dir)
    # print(main_dir + "/" + inner_dir)
    if not os.path.exists(main_dir + "/" + inner_dir):
        os.mkdir(main_dir + "/" + inner_dir)


def save_data(dir_name, clf=None, X_train=None, X_test=None, y_train=None, y_test=None):
    # save the model & train set & test set
    if X_train is not None:
        pickle.dump(X_train, open(dir_name + "/" + "X_train", 'wb'))
    if X_test is not None:
        pickle.dump(X_test, open(dir_name + "/" + "X_test", 'wb'))
    if y_test is not None:
        pickle.dump(y_test, open(dir_name + "/" + "y_test", 'wb'))
    if y_train is not None:
        pickle.dump(y_train, open(dir_name + "/" + "y_train", 'wb'))
    if clf is not None:
        joblib.dump(clf, dir_name + "/" + "clf.joblib")


def load_data(dir_name):
    # load the model & train set & test set
    clf = joblib.load(dir_name + "/clf.joblib")
    X_train = pickle.load(open(dir_name + "/" + "X_train", 'rb'))
    X_test = pickle.load(open(dir_name + "/" + "X_test", 'rb'))
    y_train = pickle.load(open(dir_name + "/" + "y_train", 'rb'))
    y_test = pickle.load(open(dir_name + "/" + "y_test", 'rb'))
    return clf, X_train, X_test, y_train, y_test


# </editor-fold>


if __name__ == '__main__':
    cluster_path = "muscle-formation-diff"
    local_path = ".."
    path = local_path
    window_size = 40
    n = 10
    l = 200

    int_measures_s5 = get_intensity_measures_df(
        csv_path=path + r"/data/mastodon/train/Nuclei_5-vertices.csv",
        video_actin_path=path + r"/data/videos/train/S5_Actin.tif",
        window_size=window_size)
    int_measures_s5.to_csv(path + "/data/mastodon/train/int_measures_s5.csv")

    int_measures_s1 = get_intensity_measures_df(
        csv_path=path + r"/data/mastodon/train/Nuclei_1-vertices.csv",
        video_actin_path=path + r"/data/videos/train/S1_Actin.tif",
        window_size=window_size)
    int_measures_s1.to_csv(path + "/data/mastodon/train/int_measures_s1.csv")

    int_measures_s2 = get_intensity_measures_df(
        csv_path=path + r"/data/mastodon/test/Nuclei_2-vertices.csv",
        video_actin_path=path + r"/data/videos/test/S2_Actin.tif",
        window_size=window_size)
    int_measures_s2.to_csv(path + "/data/mastodon/test/int_measures_s2.csv")

    int_measures_s3 = get_intensity_measures_df(
        csv_path=path + r"/data/mastodon/test/Nuclei_3-vertices.csv",
        video_actin_path=path + r"/data/videos/test/S3_Actin.tif",
        window_size=window_size)
    int_measures_s3.to_csv(path + "/data/mastodon/test/int_measures_s3.csv")

    dif_window = [200, 230]
    con_windows = [[0, 30], [140, 170], [180, 210], [240, 270], [300, 330]]

    dir_name = f"15-02-2022-manual_mastodon_motility-False_intensity-True"
    second_dir = f"{dif_window[0]},{dif_window[1]} frames ERK, {con_windows} frames con,{window_size} winsize"
    open_dirs(dir_name, second_dir)
    dir_name += "/" + second_dir

    X_train, y_train = prep_data(diff_df=int_measures_s5, con_df=int_measures_s1, diff_t_window=dif_window,
                                 con_t_windows=con_windows)
    X_test, y_test = prep_data(diff_df=int_measures_s3, con_df=int_measures_s2, diff_t_window=dif_window,
                               con_t_windows=con_windows)

    extracted_features = extract_features(X_train, column_id="Spot track ID", column_sort="Spot frame",
                                          show_warnings=False)
    impute(extracted_features)
    features_filtered = select_features(extracted_features, y_train, show_warnings=False)
    X_train = extract_relevant_features(X_train, y_train, column_id="Spot track ID", column_sort='Spot frame',
                                        show_warnings=False)

    extracted_features = extract_features(X_test, column_id="Spot track ID", column_sort="Spot frame",
                                          show_warnings=False)
    impute(extracted_features)
    X_test = extracted_features[X_train.columns]

    # train the classifier
    clf = train(X_train, y_train)
    save_data(dir_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, clf=clf)
    save_data(dir_name, clf=clf)

    del (X_train)
    del (y_train)

    report, auc_score = evaluate(clf, X_test, y_test)

    # load the model & train set & test set
    clf, X_train, X_test, y_train, y_test = load_data(dir_name)

    # plot ROC curve
    plot_roc(clf=clf, X_test=X_test, y_test=y_test, path=dir_name)

    # perform PCA analysis
    principal_df, pca = build_pca(3, X_test)
    plot_pca(principal_df, pca, dir_name)

    # calculate feature importance
    feature_importance(clf, X_train.columns, dir_name)

    # save classification report & AUC score
    txt_file = open(dir_name + '/info.txt', 'a')
    txt_file.write(f"classification report: {report}\n auc score: {auc_score}")
    txt_file.close()
