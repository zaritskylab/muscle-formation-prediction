from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import shap
import os
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame")


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


def plot_pca(principal_df, path):
    '''
    Plot and save a PCA of 3 dimensions (x axis, y axis & color)
    :param principal_df: principal components dataframe to plot
    :param path: path of the output directory
    :return: -
    '''
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="principal component 1", y="principal component 2",
        hue='target',
        palette=sns.color_palette("hls", len(principal_df['target'].unique())),
        data=principal_df,
        legend='brief',
        alpha=0.3
    )
    plt.title("PCA plot")
    plt.savefig(path + "/PCA.png")
    plt.show()

def plot_roc(clf, X_test, y_test, path):
    plt.figure(figsize=(20, 6))
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


def plot_features_boxplot(X_extracted, f_list):
    '''
    Plot all features data as boxplots
    :param X_extracted: data to plot
    :param f_list: list of features to plot
    :return: -
    '''
    for ind, col in enumerate(f_list):
        ax = sns.boxplot(x="target", y=str(col), hue="target",
                         data=X_extracted, palette="Set3")
        plt.title(col)
        plt.show()

def feature_importance(clf, feature_names, path):
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    sorted_idx = clf.feature_importances_.argsort()

    ax.barh(feature_names[sorted_idx], clf.feature_importances_[sorted_idx])
    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=50)

    plt.xlabel("Random Forest Feature Importance")
    plt.title('Feature Importance Plot')
    plt.savefig(path + "/feature importance.png")
    plt.show()

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

def get_shap_explainations(model, data):
    '''
    Plot SHAP's output explanations.
    "SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output
    of any machine learning model. It connects optimal credit allocation with local explanations
    using the classic Shapley values from game theory and their related extensions."
    https://github.com/slundberg/shap
    :param model: the model to explain
    :param data: data to explain
    :return:-
    '''
    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

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

    shap.summary_plot(shap_values, data)
    #
    # shap.dependence_plot(
    #     ("Age", "Sex"),
    #     shap_interaction_values, X.iloc[:2000,:],
    #     display_features=X_display.iloc[:2000,:]
    # )


if __name__ == '__main__':
    print("Let's go!")

    # TODO: 2. shap feature importance
    # TODO: 3. get velocity, direction mof movement, monotonicity - over the probability to differentiate
    # TODO: 4. compare #3 values for several cells with different differentiation probability.
