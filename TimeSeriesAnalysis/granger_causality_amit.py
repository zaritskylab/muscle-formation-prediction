import warnings
from pathlib import Path

from copy import deepcopy
import numpy as np
import pandas as pd
import math
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
import os
import json

from itertools import permutations
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import combinations

# import logging
# logger = logging.getLogger('my_module_name')
warnings.filterwarnings("ignore")

TEST = 'ssr_chi2test'
MAXLAG = 10
NEIGH_RADIUS = 400
NEIGH_RADIUS_RATIO = 0.95


def load_file(path):
    df = pd.read_csv(path, usecols=['frame_no', 'cell_id', 'heading', 'x', 'y'])
    return df


def kpss_test(feature_series):
    statistic, p_value, n_lags, critical_values = kpss(feature_series.values, nlags="legacy")

    # print(f'KPSS Statistic: {statistic}')
    # print(f'p-value: {p_value}')
    # print(f'num lags: {n_lags}')
    # print('Critial Values:')
    # for key, value in critical_values.items():
    #     print(f'   {key} : {value}')
    return p_value


def adf_test(feature_series):
    result = adfuller(feature_series.values)
    # print('ADF Statistics: %f' % result[0])
    # print('p-value: %f' % result[1])
    # print('Critical values:')
    # for key, value in result[4].items():
    #     print('\t%s: %.3f' % (key, value))
    return result[1]


def stationary_test(cell_trajectory):
    # print(f"** start test stationary - cell_id: {cell_id} **")
    is_stat_per_feature = []
    p_value_adf = adf_test(cell_trajectory)
    p_value_kpss = kpss_test(cell_trajectory)
    is_stat_per_feature.append(True if p_value_adf <= 0.05 or p_value_kpss >= 0.05 else False)
    # cell_trajectory.to_csv(f'./stat_files/{all(is_stat_per_feature)}_stat_cell_{cell_id}.csv')
    return all(is_stat_per_feature)


def granger_causality_matrix(data, features, neigh_df, res_df, res_col, weight_param, test='ssr_chi2test',
                             maxlag=MAXLAG,
                             verbose=False, warm_up_window=0, neigh_radius=10, neigh_radius_ratio=NEIGH_RADIUS_RATIO):
    """
        The row are the response (y) and the columns are the predictors (x)
        If a given p-value is < significance level (0.05), we can reject the null hypothesis and conclude that walmart_x Granger causes apple_y.
    """
    for col in features:
        print(f"** Start calculate GC on {col} **")
        time_series_per_cell = {}
        count_non_stat = 0
        feature_data = data[['frame_no', 'cell_id', col]].copy()
        for cell_id in feature_data['cell_id'].unique():
            delta_values = feature_data[feature_data['cell_id'] == cell_id].iloc[
                           warm_up_window:].set_index('frame_no')[col].diff()[1:]
            ## todo delta_values_fitted
            delta_values_fitted = delta_values.apply(
                lambda dv: dv if np.abs(dv) < np.abs(dv - (2 * np.pi)) else dv - (2 * np.pi))
            delta_values_fitted = delta_values
            is_stat = stationary_test(delta_values_fitted.reset_index(drop=True, inplace=False))
            if not is_stat:
                count_non_stat += 1
                print(f'{cell_id} is not stationary')
                continue
            time_series_per_cell[cell_id] = delta_values_fitted
        neigh_pairs = check_neighbors(neigh_df, time_series_per_cell.keys(), neigh_radius, neigh_radius_ratio)
        time_frame_df = pd.DataFrame(time_series_per_cell)
        # for p, r in permutations(res_df[col][0][weight_param].index, 2):
        all_pairs = []
        for p, r in permutations(feature_data['cell_id'].unique(), 2):
            all_pairs.extend([(p, r), (r, p)])
        for p, r in all_pairs:
            ## r = response: the follower
            ## p = predictor: the leader
            if p not in time_frame_df or r not in time_frame_df or \
                    ((p, r) not in neigh_pairs and (r, p) not in neigh_pairs):
                continue
            pair_cells = time_frame_df[[r, p]].dropna()
            model = VAR(pair_cells)
            adjusted_lag = maxlag
            while True:
                try:
                    lags_results = model.select_order(adjusted_lag)
                    break
                except np.linalg.LinAlgError as err:
                    adjusted_lag -= 1
            lags = [lags_results.aic, lags_results.bic]
            opt_lag = np.min(lags)
            ## if the minimum is 0, the maximum will be taken. if it also 0, 1 will be taken.
            if opt_lag == 0:
                print(f"at least one if the metrics yield lag 0")
                opt_lag = np.max([np.max(lags), 1])  ## TODO change the 1
                if np.max(lags) == 0:
                    print(f"both lags are 0; 1 will be taken")
            gc_result = grangercausalitytests(pair_cells, maxlag=opt_lag, verbose=verbose)
            p_value = gc_result[opt_lag][0][test][1]
            res_df[col][res_col][0].loc[int(p), int(r)] = p_value
        res_df[col][res_col][1] = count_non_stat / len(feature_data['cell_id'].unique())
    return res_df


def check_neighbors(neigh_df, cells_ids, neigh_radius, neigh_dist_ratio, verbose=False):
    neigh_list = []
    all_pairs = list(combinations(list(cells_ids), 2))
    for c1, c2 in all_pairs:
        c1_pos_df = neigh_df[neigh_df['cell_id'] == c1].set_index('frame_no')
        c2_pos_df = neigh_df[neigh_df['cell_id'] == c2].set_index('frame_no')
        merge_df = pd.concat([c1_pos_df.add_prefix('c1_'), c2_pos_df.add_prefix('c2_')], axis=1).dropna()
        # dist_df = c1_pos_df.apply(lambda c1_pos: math.dist(c1_pos[['x', 'y']].values,
        #                                          c2_pos_df[c2_pos_df['frame_no'] == c1_pos['frame_no']].iloc[0][
        #                                              ['x', 'y']].values), axis=1)
        dist_df = merge_df.apply(lambda pos: math.dist(pos[['c1_x', 'c1_y']].values, pos[['c2_x', 'c2_y']].values),
                                 axis=1)
        is_neigh = dist_df < neigh_radius
        if np.mean(is_neigh) >= neigh_dist_ratio:
            neigh_list.append((c1, c2))
        else:
            if verbose:
                print(f'neighbors test: {c1} and {c2} are neighbors only on {np.mean(is_neigh)}% of the frames')
    return neigh_list


def expand_weights_vector(weights_vector, num_of_cells):
    expanded_weights = np.full(num_of_cells - len(weights_vector), fill_value=weights_vector[-1], dtype=np.float64)
    expanded_weights = weights_vector + list(expanded_weights)
    cell_ids = []
    most_dominant = str(num_of_cells - 1 - np.argmax(expanded_weights))
    dominant_cells = [most_dominant]
    for i, val in enumerate(expanded_weights):
        if val > 0:
            cell_ids.append(str(num_of_cells - 1 - i))
            if val > 0.5:
                dominant_cells.append(str(num_of_cells - 1 - i))
    return cell_ids, dominant_cells


def extract_interacted_cells(sim_params_dict):
    num_of_cells = int((sim_params_dict['length'] ** 2) * sim_params_dict['density'])

    leaders_weights = sim_params_dict['leader_weights']
    leaders_ids, dominant_leaders = expand_weights_vector(leaders_weights, num_of_cells)

    followers_weights = sim_params_dict['follower_weights']
    followers_ids, dominant_followers = expand_weights_vector(followers_weights, num_of_cells)

    return leaders_ids, followers_ids, dominant_leaders, dominant_followers


def init_gc_df(all_simulation_gc_df, cell_ids, columns, sim_name):
    cell_ids_int = [int(c) for c in cell_ids]
    gc_df_all = pd.DataFrame(np.full((len(cell_ids), len(cell_ids)), np.nan), columns=cell_ids_int, index=cell_ids_int)
    for col in set(columns) - set(['frame_no', 'cell_id']):
        if col not in all_simulation_gc_df:
            all_simulation_gc_df[col] = {}
        all_simulation_gc_df[col][sim_name] = [gc_df_all.copy(), np.nan]
    return all_simulation_gc_df


def run(paths_list=None, top_folder='', warm_up_window=0, separate_outputs=False, neigh_radius_ratio=NEIGH_RADIUS_RATIO,
        maxlag=MAXLAG, neigh_radius=NEIGH_RADIUS):
    if separate_outputs:
        separate_output_dict = {0: 'leader', 1: 'follower', 2: 'control'}
    else:
        separate_output_dict = {0: 'all'}
    if paths_list is None:
        root_path = os.path.dirname(os.path.realpath(__file__))
        files_path = Path(root_path) / 'examples' / top_folder
        paths_list = files_path.rglob('*.csv')
        try:
            simulation_name = top_folder[top_folder.rindex('/') + 1:]
        except:
            simulation_name = top_folder
    else:
        paths_list = [Path(p) for p in paths_list]
    all_simulation_gc_df = {}
    for simulation_no, path in enumerate(paths_list):
        filename = path.name
        files_path = path.parent
        folder = path.parent.parts[-1]
        if not top_folder:
            simulation_name = filename[:-4]
        print(f'start handling file no. {simulation_no}: {folder}/{filename}')
        weight_param = folder[folder.rindex('_') + 1:]
        df = load_file(path)
        df['cell_id'] = df['cell_id'].astype(str)
        neigh_df = df[['cell_id', 'frame_no', 'x', 'y']].copy()
        df = df.drop(columns=['x', 'y'])
        cell_ids = df['cell_id'].unique()
        all_simulation_gc_df = init_gc_df(all_simulation_gc_df, cell_ids, df.columns, sim_name=weight_param)
        granger_causality_matrix(df, features=set(df.columns) - set(['frame_no', 'cell_id']), neigh_df=neigh_df,
                                 maxlag=maxlag,
                                 res_df=all_simulation_gc_df, res_col=weight_param, weight_param=weight_param,
                                 warm_up_window=warm_up_window, neigh_radius=neigh_radius,
                                 neigh_radius_ratio=neigh_radius_ratio)

    for col, sim_results_instances in all_simulation_gc_df.items():
        create_multiple_gc_plot(sim_results_instances, files_path, col, simulation_name, is_all=True)


def create_multiple_gc_plot(dfs_dict, files_path, col, simulation_name, is_all):
    no_simulations = len(dfs_dict)
    no_cells = list(dfs_dict.values())[0][0].shape[0]
    width = no_simulations * no_cells * 5
    height = no_cells * 6
    fig_, subfigs = plt.subplots(1, len(dfs_dict), sharey=True, figsize=(width, height), squeeze=False)

    count_figures = 0
    for df_name, (df_all, stat_ratio) in dfs_dict.items():
        # leaders_ids, followers_ids, dominant_leaders, dominant_followers = extract_interacted_cells(sim_params)
        # df_all.to_csv(f'{files_path}/gc_{col}_{df_name}_all.csv')
        create_gc_heatmap(subfigs[0][count_figures], df_all, f'{files_path}/gc_{col}_{df_name}_all', simulation_name,
                          stat_ratio, count_figures == len(dfs_dict) - 1, is_all)
        subfigs[0][count_figures].set_xlabel(df_name, fontsize='x-large')
        count_figures += 1

    fig_.suptitle(f'GC p-value {col} - {simulation_name}', fontsize='xx-large')
    fig_.supxlabel('follower', fontsize='xx-large')
    fig_.supylabel('leaders', fontsize='xx-large')
    plt.savefig(f'{files_path}/{simulation_name}_gc_{col}_all.jpg')
    plt.close()


def create_gc_heatmap(ax, df, output_path, simulation_name, stat_ratio, cbar, is_all):
    if not is_all:
        fig, ax = plt.subplots(figsize=(70, 90))  # Sample figsize in inches
        ax.set_title(f'GC p-value - {simulation_name} \n passed the stationary test - {stat_ratio * 100}%\n',
                     fontdict={'fontsize': 60})
    else:
        ax.set_title(f'passed the stationary test - {stat_ratio * 100}%', fontdict={'fontsize': 60})
    plt.rc('font', size=40)  # controls default text sizes
    plt.rc('legend', fontsize=50)  # legend fontsize
    sns.heatmap(df.iloc[::-1], annot=True, ax=ax, vmin=0.05, vmax=0.4, cbar=cbar)
    if not is_all:
        ax.set_ylabel('cells pairs', fontsize=60)
        ax.set_xlabel('follower weight', fontsize=60)
    # add_bold_lines(ax, dominant_followers, number_of_cells=df.shape[0], orientation='vertical', color='red')
    # add_bold_lines(ax, dominant_leaders, number_of_cells=df.shape[0], orientation='horizon', color='green')
    ax.tick_params(axis='y', size=15, rotation=0, labelsize=50)
    ax.tick_params(axis='x', size=15, labelsize=50)
    # plt.savefig(f'{output_path}.jpg')
    # plt.close()


def add_bold_lines(ax, dominant_cells, number_of_cells, orientation, color):
    def _draw_hlive(y):
        ax.hlines(y=y, xmin=0, xmax=number_of_cells, linestyles='solid', colors=color, lw=10)

    def _draw_vline(x):
        ax.vlines(x=x, ymin=0, ymax=number_of_cells, linestyles='solid', colors=color, lw=10)

    for cell_id in dominant_cells:
        if orientation == 'horizon':
            _draw_hlive(number_of_cells - int(cell_id) - 1)
            _draw_hlive(number_of_cells - int(cell_id))
        elif orientation == 'vertical':
            _draw_vline(int(cell_id))
            _draw_vline(int(cell_id) + 1)


if __name__ == '__main__':
    run(paths_list=['../data/gramger_cos/s3_g1_g2_g3.csv'], warm_up_window=0,
        separate_outputs=True, neigh_radius_ratio=NEIGH_RADIUS_RATIO, maxlag=MAXLAG,
        neigh_radius=NEIGH_RADIUS)

    ## if you want to process a folder and then to get all plots in the same figure use top_folder
    # run(top_folder='amit_data', warm_up_window=0,
    #     separate_outputs=True,neigh_radius_ratio=NEIGH_RADIUS_RATIO, maxlag=MAXLAG,
    #     neigh_radius=NEIGH_RADIUS)
