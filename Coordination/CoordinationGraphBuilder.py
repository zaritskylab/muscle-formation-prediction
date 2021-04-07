# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:52:27 2020

@author: Oron & Amit Shakarchy
"""
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Scripts.DataPreprocessing.load_tracks_xml import load_tracks_xml
import pandas as pd


class CoordGraphBuilder():

    def __init__(self, coord_control_path=None, coord_diff_path=None, control_xml_path=None, diff_xml_path=None):
        if coord_control_path is None or coord_diff_path is None or control_xml_path is None or diff_xml_path is None:
            pass
        else:
            self.coord_diff = pickle.load(open(coord_diff_path, 'rb'))
            self.coord_control = pickle.load(open(coord_control_path, 'rb'))
            self.control_xml_path = control_xml_path
            self.diff_xml_path = diff_xml_path

    def read_coordinationDF(self, coordDF):
        K = coordDF.shape[0]
        ar = np.zeros((K, 927))
        ar[:] = np.nan
        time = np.zeros((K, 927))
        time[:] = np.nan
        for i in range(K):
            t0 = int(coordDF['t0'].iloc[i])
            cost = coordDF['cos_theta'].iloc[i]
            time_ = coordDF['t0'].iloc[i]
            N = len(cost)
            ar[i, t0:t0 + N] = cost
            time[i, t0:t0 + N] = time_ * 1.5 / 60
        meanar03 = np.nanmean(ar, axis=0)
        mean_time = np.nanmean(time, axis=0)
        return meanar03, mean_time

    def plot_coord_over_time(self, name_for_saving):
        # Read coordination DFs
        control, time_c = self.read_coordinationDF(self.coord_control)
        diff, time_d = self.read_coordinationDF(self.coord_diff)
        # Plot
        fig = plt.figure(figsize=(6, 4))
        plt.plot(control[:-7], )
        plt.plot(diff[:-7], )
        plt.legend(['Control', 'Diff'])
        plt.title(r'Cos of $\theta$ measure')
        plt.xticks(np.arange(0, 921, 100), labels=np.around(np.arange(0, 921, 100) * 1.5 / 60, decimals=1))
        # plt.yticks(np.arange(0.45, 1, 0.05))
        plt.ylim(0.6, 1, 0.5)
        plt.xlabel('Time [h]')
        plt.ylabel(r'Cos of $\theta$')
        plt.savefig(name_for_saving)
        plt.show()
        plt.close(fig)

    def get_density(self, tracks_xml):
        tracks, df = load_tracks_xml(tracks_xml)
        s01den = np.empty((len(tracks), 926))
        for k, track in enumerate(tracks):
            start = int(np.min(np.asarray(track['t_stamp'])))
            stop = int(np.max(np.asarray(track['t_stamp'])))
            s01den[k, start:stop] = 1
        s01out = np.sum(s01den, axis=0)
        return s01out

    def get_coord_density_df(self, coord_df, coord_xml_path):
        # Read coordination DFs
        coord, time = self.read_coordinationDF(coord_df)
        # Get density from xml
        density = self.get_density(coord_xml_path)
        # Attach to one DF
        df = pd.DataFrame({'density': pd.DataFrame(density)[0][:920],
                           'coordination': pd.DataFrame(coord)[0][:920],
                           'time': pd.DataFrame(time)[0][:920]})
        return df

    def plot_coor_over_density_all_exp(self, name_for_saving, to_plot):
        data = []
        if to_plot == "control":
            ind = (1, 2, 7, 8, 9, 10)
        else:
            ind = (3, 4, 5, 6, 11, 12)  # , 6, 11, 12
        for i in ind:
            coord_path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\coherence_polinomialFit_s{}".format(
                i)
            coord_xml = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Single cell\Tracks_xml\Experiment1_w1Widefield550_s{}_all.xml".format(
                i)
            builder = CoordGraphBuilder(coord_path, coord_path, coord_xml, coord_xml)
            df = self.get_coord_density_df(builder.coord_diff, builder.diff_xml_path)
            data.append([i, df["density"][0], df["coordination"][0]])

        all_df = pd.DataFrame(data, columns=['i', "density", 'coordination'])
        all_df = all_df.sort_values("density")
        n = all_df["density"]
        s = all_df["coordination"]
        plt.bar(n, s, color="cornflowerblue")
        plt.plot(n, s, color="blue")
        plt.title(r'Coordination over density- all experiments')
        plt.grid()
        plt.xlabel('Density')
        plt.ylabel(r'Cos of $\theta$')
        plt.xticks(all_df["density"])
        plt.ylim(0.5, 1, 0.25)
        for i in range(len(s)):
            plt.annotate("exp#{}".format(all_df["i"][i]), xy=(n[i], s[i]), ha='center', va='bottom', fontsize=6)
        plt.savefig(name_for_saving)
        plt.show()

    def plot_coor_over_density(self, name_for_saving):
        control_den_df = self.get_coord_density_df(self.coord_control, self.control_xml_path)
        diff_den_df = self.get_coord_density_df(self.coord_diff, self.diff_xml_path)

        fig = plt.figure(figsize=(8, 4))

        GnBu_rBig = cm.get_cmap('Blues_r', 512)
        GnBu = ListedColormap(GnBu_rBig(np.linspace(0.25, 0.75, 256)))
        RdPu_rBig = cm.get_cmap('Oranges_r', 512)
        RdPu = ListedColormap(RdPu_rBig(np.linspace(0.25, 0.75, 256)))

        plt.scatter(x=control_den_df["density"], y=control_den_df["coordination"], marker=".", color="royalblue",
                    label="Control")
        control_scatter = plt.scatter(x=control_den_df["density"], y=control_den_df["coordination"],
                                      c=control_den_df["time"], marker=".", cmap=GnBu)
        plt.scatter(x=diff_den_df["density"], y=diff_den_df["coordination"], marker=".", color="orange", label="Diff")
        dif_scatter = plt.scatter(x=diff_den_df["density"], y=diff_den_df["coordination"], c=diff_den_df["time"],
                                  marker=".", cmap=RdPu)

        c_bar = plt.colorbar(control_scatter, shrink=0.6)
        c_bar.set_label('time (h), control')
        d_bar = plt.colorbar(dif_scatter, shrink=0.6)
        d_bar.set_label('time (h), diff')
        plt.legend()
        plt.xticks(np.arange(50, 450, 50))
        plt.ylim(0.45, 1, 0.5)
        plt.title(r'Cos of $\thet'
                  r'a$ measure (Coordination over Density)')
        plt.xlabel('Density')
        plt.ylabel(r'Cos of $\theta$')
        plt.savefig(name_for_saving)
        plt.show()
        plt.close(fig)

    def plot_validation(self, name_for_saving, color):
        # Read coordination DFs
        control, _ = self.read_coordinationDF(self.coord_control)
        diff, _ = self.read_coordinationDF(self.coord_diff)
        coord_permute_diff = pickle.load(
            open(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\coherence_validaton_3',
                 'rb'))
        coord_permute_con = pickle.load(
            open(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\coherence_validaton_1',
                 'rb'))
        coord_all_random_diff = pickle.load(
            open(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\all_random_angles_video_3',
                 'rb'))
        coord_all_random_con = pickle.load(
            open(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\all_random_angles_video_1',
                 'rb'))
        coord_all_random_total = pickle.load(
            open(
                r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\all_random_angles_video_total_random',
                'rb'))

        total_random, _ = self.read_coordinationDF(coord_all_random_total)
        all_random_diff, _ = self.read_coordinationDF(coord_all_random_diff)
        all_random_con, _ = self.read_coordinationDF(coord_all_random_con)
        permute_diff, _ = self.read_coordinationDF(coord_permute_diff)
        permute_con, _ = self.read_coordinationDF(coord_permute_con)
        fit_diff = np.polyval(np.polyfit(np.arange(920), diff[:-7], deg=2), np.arange(920))
        fit_con = np.polyval(np.polyfit(np.arange(920), control[:-7], deg=2), np.arange(920))
        fit_permute_diff = np.polyval(np.polyfit(np.arange(920), permute_diff[:-7], deg=2), np.arange(920))
        fit_permute_con = np.polyval(np.polyfit(np.arange(920), permute_con[:-7], deg=2), np.arange(920))

        # Plot
        fig = plt.figure(figsize=(6, 4))

        # control
        plt.plot(pd.DataFrame(control[:-7], columns=["control"]).rolling(window=10).mean(), color="blue")
        plt.plot(pd.DataFrame(permute_con[:-7], columns=["permute_con"]).rolling(window=10).mean(), '--', color="blue")

        # diff
        plt.plot(pd.DataFrame(diff[:-7], columns=["diff"]).rolling(window=10).mean(), color="orange")
        plt.plot(pd.DataFrame(permute_diff[:-7], columns=["permute_diff"]).rolling(window=10).mean(), '--',
                 color="orange")

        plt.plot(pd.DataFrame(total_random[:-7], columns=["permute_diff"]).rolling(window=10).mean(), '--',
                 color="red")

        plt.legend(['Control - local',
                    'Control - global',
                    'ERKi treatment - local',
                    'ERKi treatment - global',
                    'Random'])
        plt.title(r'Coordination over time')
        plt.xticks(np.arange(0, 921, 100), labels=np.around(np.arange(0, 921, 100) * 1.5 / 60, decimals=1))
        # plt.yticks(np.arange(0.45, 1, 0.05))
        # plt.ylim(0.6, 1, 0.25)
        plt.xlabel('Time [h]')
        plt.ylabel(r'Coordination')
        plt.savefig('Coordination_over_time.eps', format='eps')
        plt.show()

        # Read coordination DFs
        original, _ = self.read_coordinationDF(self.coord_control)
        permute, _ = self.read_coordinationDF(self.coord_diff)

        if color == "blue":
            c_original = "blue"
            c_permute = "cornflowerblue"
        else:
            c_original = "orange"
            c_permute = "navajowhite"
        # Plot
        fig = plt.figure(figsize=(6, 4))
        plt.plot(original[:-7], color=c_original)
        plt.plot(permute[:-7], color=c_permute)
        plt.legend(['Control', 'Diff'])
        plt.title(r'Cos of $\theta$ measure')
        plt.xticks(np.arange(0, 921, 100), labels=np.around(np.arange(0, 921, 100) * 1.5 / 60, decimals=1))
        # plt.yticks(np.arange(0.45, 1, 0.05))
        plt.ylim(0.6, 1, 0.5)
        plt.xlabel('Time [h]')
        plt.ylabel(r'Cos of $\theta$')
        plt.savefig(name_for_saving)
        plt.show()
        plt.close(fig)

    def plot_coor_over_time_normalized(self, name_for_saving):
        # Read coordination DFs
        control, time_c = self.read_coordinationDF(self.coord_control)
        diff, time_d = self.read_coordinationDF(self.coord_diff)

        fit01 = np.polyval(np.polyfit(np.arange(920), control[:-7], deg=2), np.arange(920))
        fit03 = np.polyval(np.polyfit(np.arange(920), diff[:-7], deg=2), np.arange(920))

        # Plot
        fig = plt.figure(figsize=(6, 4))
        plt.plot(fit01, diff[:-7], color="orange")
        plt.gca().invert_xaxis()
        plt.legend(['Diff'])
        plt.title(r'Cos of $\theta$ measure')
        plt.xlabel('Control\'s fit line')
        plt.ylabel(r'Cos of $\theta$')
        plt.savefig(name_for_saving)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    con_i = 10
    dif_i = 12

    # Load coordination_outputs dataframes
    coord_control_path = r'coordination_outputs/validations/pickled coordination dfs/0104_regular/coordination_df_s{}_0104.pkl'.format(con_i)
    coord_diff_path = r'coordination_outputs/validations/pickled coordination dfs/0104_regular/coordination_df_s{}_0104.pkl'.format(dif_i)

    xml_control_path = r'../../data\Tracks_xml\0104\Experiment1_w1Widefield550_s{}_all_0104.xml'.format(con_i)
    xml_diff_path = r'../../data\Tracks_xml\0104\Experiment1_w1Widefield550_s{}_all_0104.xml'.format(dif_i)

    builder = CoordGraphBuilder(coord_control_path, coord_diff_path, xml_control_path, xml_diff_path)
    # builder.plot_validation("save","blue")
    builder.plot_coor_over_density("Coordination Over Density (Exp {},{})".format(con_i,dif_i))
    builder.plot_coord_over_time("Coordination Over Time (Exp {},{})".format(con_i,dif_i))
