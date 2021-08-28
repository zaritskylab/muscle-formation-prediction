from aae.AELoader import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from bioinfokit.visuz import cluster
import numpy as np
import pandas as pd
import seaborn as sns


def pca_scree_plot(encoded_imgs):
    '''
    Creates PCA's scree plot.
    :param encoded_imgs: encoded images
    :return: saves a scree plot
    '''
    # build PCA's components
    pca_out = PCA().fit(encoded_imgs)
    num_pc = pca_out.n_features_
    pc_list = ["PC" + str(i) for i in list(range(1, num_pc + 1))]
    cluster.screeplot(obj=[pc_list, pca_out.explained_variance_ratio_])


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


def plot_pca(principal_df, title):
    '''
    The method plots the first 3 dimensions of a given PCA
    :param principal_df: PCA dataframe
    :return: no return value
    '''
    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=10)
    # ax.set_ylabel('Principal Component 2', fontsize=10)
    # ax.set_title('2 component PCA', fontsize=15)
    # ax.scatter(principal_df['principal component 1'],
    #            principal_df['principal component 2'],
    #            principal_df['principal component 3'])
    # ax.grid()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue='principal component 3',
        palette=sns.color_palette("hls", len(principal_df['principal component 3'].unique())),
        data=principal_df,
        legend=False,
        alpha=0.3
    )
    plt.title(title)
    plt.show()


def plot_pca_on_time(principal_df, time_df):
    '''
    The method merges the PCA dataframe and the matching time dataframe
    and plots PCA graph with time as the color dimension
    :param principal_df: PCA graph
    :param time_df: matching time dataframe
    :return: the merged df
    '''
    principal_df = pd.merge(principal_df, time_df, left_index=True, right_index=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue='Time',
        palette=sns.color_palette("hls", len(principal_df['Time'].unique())),
        data=principal_df,
        legend=False,
        alpha=0.3
    )
    plt.show()
    return principal_df


def plot_original_imgs(indexes, imgs):
    '''
    The method displays 20 original images located in the needed indexes.
    :param indexes: list of indexes to take images from
    :param imgs: nd-array hold the images
    :return:
    '''
    # Plot the original images from a specific cluster
    num_of_images = 20  # Display 20 images
    plt.figure(figsize=(8, 4))
    for i in range(num_of_images):
        if len(indexes) > i:
            ax = plt.subplot(3, num_of_images, i + 1)

            # Get the original image from the 'new_test' dataset, using its index on the PCA graph
            plt.imshow(imgs[indexes[i]])

            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


if __name__ == '__main__':
    encoder = load_model("models/encoder_test")
    imgs = np.load("../data/test_32.npy")[:100]
    encoded_imgs = encoder.predict(imgs)
    principalDf = build_pca(num_of_components=3, encoded_images=encoded_imgs)
    cluster1_indexes = principalDf.index[(principalDf['principal component 1'] > -5) &
                                         (principalDf['principal component 2'] < 0)].tolist()
    plot_pca(principal_df=principalDf)
    plot_original_imgs(cluster1_indexes, imgs)
