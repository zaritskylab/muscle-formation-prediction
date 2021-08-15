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
from DataPreprocessing.load_tracks_xml import load_tracks_xml
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
            time[i, t0:t0 + N] = time_ * 5 / 60
        meanar03 = np.nanmean(ar, axis=0)
        mean_time = np.nanmean(time, axis=0)
        return meanar03, mean_time

    def get_mean_coordination(self, ind1, ind2):
        elements_num = 2
        coord_length = 927
        pickle_path = r'coordination_outputs/coordination_dfs/different_densities/small field of view/coordination_df_s{}_30_small.pkl'
        total_coord = np.zeros(shape=(coord_length,))
        total_time = np.zeros(shape=(coord_length,))
        for i in (ind1, ind2):
            df = pickle.load(
                open(pickle_path.format(i), 'rb'))
            coord, time = self.read_coordinationDF(df)
            total_time += time
            total_coord += coord
        return total_coord / elements_num, total_time / elements_num

    def plot_coord_over_time(self, name_for_saving):

        # Read coordination DFs
        control, _ = self.read_coordinationDF(self.coord_control)
        diff, _ = self.read_coordinationDF(self.coord_diff)

        # Plot
        fig = plt.figure(figsize=(6, 4))
        plt.plot(pd.DataFrame(control[:920], columns=["permute_diff"]).rolling(window=10).mean(), )
        plt.plot(pd.DataFrame(diff[:920], columns=["permute_diff"]).rolling(window=10).mean(), )
        plt.legend(['Controll', 'differentiation', ])
        plt.title(r'Cos of $\theta$ measure')
        plt.xticks(np.arange(0, 920, 100), labels=np.around(np.arange(0, 920, 100) * 1.5 / 60, decimals=1))
        # plt.yticks(np.arange(0.45, 1, 0.05))
        plt.ylim(0.6, 1, 0.5)
        plt.xlabel('Time [h]')
        plt.ylabel(r'Cos of $\theta$')
        plt.savefig(name_for_saving)
        plt.show()
        plt.close(fig)

    def plot_coord_over_time_4_measurements(self, name_for_saving):

        coord_control, _ = self.get_mean_coordination(1, 2, 3)
        coord_diff_fusion, _ = self.get_mean_coordination(4, 5, 6)
        coord_proliferation, _ = self.get_mean_coordination(7, 8, 9)
        coord_diff, _ = self.get_mean_coordination(10, 11, 12)

        # Read coordination DFs
        # control, _ = self.read_coordinationDF(self.coord_control)
        # diff, _ = self.read_coordinationDF(self.coord_diff)

        random_total = pickle.load(
            open(
                r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\all_random_angles_video_total_random',
                'rb'))

        total_random, _ = self.read_coordinationDF(random_total)
        # Plot
        fig = plt.figure(figsize=(6, 4))
        for df in (coord_control, coord_diff, coord_diff_fusion, coord_proliferation):
            plt.plot(pd.DataFrame(df[:547], columns=["permute_diff"]).rolling(window=10).mean(), )
        plt.plot(pd.DataFrame(total_random[:547], columns=["permute_diff"]).rolling(window=10).mean(), '--')

        plt.legend(['Negative control -proliferation',
                    'differentiation',
                    'differentiation and Fusion',
                    'proliferation', 'random'])
        plt.title(r'Cos of $\theta$ measure')
        plt.xticks(np.arange(0, 554, 60), labels=np.around(np.arange(0, 554, 60) * 2.5 / 60, decimals=1))
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

    def get_mean_density(self, ind1, ind2, ind3):
        elements_num = 3
        coord_length = 926
        xml_path = r"../data/tracks_xml/Adi/201209_p38iexp_live_1_Widefield550_s{}_all_8bit_ScaleTimer.xml"
        total_density = np.zeros(shape=(coord_length,))
        for i in (ind1, ind2, ind3):
            density = self.get_density(xml_path.format(i))
            total_density += density
        return total_density / elements_num

    def get_mean_density_df(self, ind1, ind2, ind3):
        mean_den = self.get_mean_density(ind1, ind2, ind3)
        mean_coord, time = self.get_mean_coordination(1, 2, 3)
        # Attach to one DF
        den_df = pd.DataFrame({'density': pd.DataFrame(mean_den)[0][:920],
                               'coordination': pd.DataFrame(mean_coord)[0][:920],
                               'time': pd.DataFrame(time)[0][:920]})
        return den_df

    def plot_coor_over_density(self, name_for_saving):

        # control_den_df = self.get_coord_density_df(self.coord_control, self.control_xml_path)
        # diff_den_df = self.get_coord_density_df(self.coord_diff, self.diff_xml_path)

        den_control = self.get_mean_density_df(1, 2, 3)
        den_diff_fusion = self.get_mean_density_df(4, 5, 6)
        den_proliferation = self.get_mean_density_df(7, 8, 9)
        den_diff = self.get_mean_density_df(10, 11, 12)

        # control_den_df = self.get_coord_density_df(con_df,
        #                                            "../data/tracks_xml/Adi/201209_p38iexp_live_1_Widefield550_s2_all_8bit_ScaleTimer.xml")
        # diff_fusion_den_df = self.get_coord_density_df(diff_fusion_df,
        #                                                "../data/tracks_xml/Adi/201209_p38iexp_live_1_Widefield550_s5_all_8bit_ScaleTimer.xml")
        # proliferation_den_df = self.get_coord_density_df(proliferation_df,
        #                                                  "../data/tracks_xml/Adi/201209_p38iexp_live_1_Widefield550_s8_all_8bit_ScaleTimer.xml")
        # diff_den_df = self.get_coord_density_df(diff_df,
        #                                         "../data/tracks_xml/Adi/201209_p38iexp_live_1_Widefield550_s11_all_8bit_ScaleTimer.xml")

        GnBu_rBig = cm.get_cmap('Blues_r', 512)
        GnBu = ListedColormap(GnBu_rBig(np.linspace(0.25, 0.75, 256)))
        RdPu_rBig = cm.get_cmap('Oranges_r', 512)
        RdPu = ListedColormap(RdPu_rBig(np.linspace(0.25, 0.75, 256)))

        fig = plt.figure(figsize=(8, 4))
        plt.scatter(x=den_control["density"], y=den_control["coordination"], marker=".", color="royalblue",
                    label="Control")
        plt.scatter(x=den_diff["density"], y=den_diff["coordination"], marker=".", color="orange", label="Diff")
        plt.scatter(x=den_diff_fusion["density"], y=den_diff_fusion["coordination"], marker=".", color="green",
                    label="Control")
        plt.scatter(x=den_proliferation["density"], y=den_proliferation["coordination"], marker=".", color="red",
                    label="Control")

        control_scatter = plt.scatter(x=den_control["density"], y=den_control["coordination"],
                                      c=den_control["time"], marker=".", cmap=GnBu)
        dif_scatter = plt.scatter(x=den_diff["density"], y=den_diff["coordination"], c=den_diff["time"],
                                  marker=".", cmap=RdPu)
        diff_fusion_scatter = plt.scatter(x=den_diff_fusion["density"], y=den_diff_fusion["coordination"],
                                          c=den_diff_fusion["time"], marker=".", cmap="Greens_r")
        proliferation_scatter = plt.scatter(x=den_proliferation["density"], y=den_proliferation["coordination"],
                                            c=den_proliferation["time"], marker=".", cmap="Reds_r")
        c_bar = plt.colorbar(control_scatter, shrink=0.6)
        c_bar.set_label('time (h), control')
        d_bar = plt.colorbar(dif_scatter, shrink=0.6)
        d_bar.set_label('time (h), diff')
        d_f_bar = plt.colorbar(diff_fusion_scatter, shrink=0.6)
        d_f_bar.set_label('time (h), diff_fusion')
        d_f_bar = plt.colorbar(proliferation_scatter, shrink=0.6)
        d_f_bar.set_label('time (h), proliferation')
        plt.legend(["control", "differentiation", "differentiation and fusion", "proliferation"])
        plt.xticks(np.arange(50, 700, 100))
        plt.ylim(0.6, 1, 0.5)
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
            open(r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\coherence_validaton_1',
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


if __name__ == '__main__':
    print()
    coord_control_path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\manual tracking\coordination_df_s3_30.pkl"
    xml_control_path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\manual tracking\Experiment1_w1Widefield550_s3_all.xml"
    builder = CoordGraphBuilder(coord_control_path, coord_control_path, xml_control_path, xml_control_path)
    builder.plot_coord_over_time("Coordination over time validation")

#     coord_control_path = r'coordination_outputs/validations/pickled coordination dfs/0104_regular/coordination_df_s3_0104.pkl'
#     xml_control_path = r'data/tracks_xml/0104/Experiment1_w1Widefield550_s1_all_0104.xml'
#     coord_validation_path = r'C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta/coherence_validaton_3'
#     xml_validation_path = r'data/tracks_xml/0104/Experiment1_w1Widefield550_s1_all_0104.xml'
#
#     # builder = CoordGraphBuilder(coord_control_path, coord_validation_path, xml_control_path, xml_validation_path)
#     # builder.plot_coord_over_time("Coordination over time validation")
#
#     # fig = plt.figure(figsize=(20, 4))
#     # colors = ["Oranges", "Blues"]
#     # for i in (1, 4, 9, 10, 12, 19, 20):
#     #     xml_path = r'../data/tracks_xml/different_densities/s{}_all.xml'.format(i)
#     #     path = r'coordination_outputs/coordination_dfs/different_densities/coordination_df_s{}.pkl'.format(i)
#     #     coord = pickle.load(open(path, 'rb'))
#     #     den_df = builder.get_coord_density_df(coord, xml_path)
#     #     plt.scatter(x=den_df["density"], y=den_df["coordination"], marker=".", label="Control")
#     #     scatter = plt.scatter(x=den_df["density"], y=den_df["coordination"],
#     #                           c=den_df["time"], marker=".")
#     #
#     #     c_bar = plt.colorbar(scatter, shrink=0.6)
#     #     c_bar.set_label('time (h), {}'.format(i))
#     #     plt.legend([1, 4, 9, 10, 12, 19, 20])
#     #     plt.xticks(np.arange(700, 3000, 500))
#     #     # plt.ylim(0.6, 1, 0.5)
#     #     plt.title(r'Cos of $\thet'
#     #               r'a$ measure (Coordination over Density)')
#     #     plt.xlabel('Density')
#     #     plt.ylabel(r'Cos of $\theta$')
#     #     plt.show()
#     #     plt.close(fig)
# fig = plt.figure(figsize=(8, 4))
# for i in ([18, 21],[11, 14], [7, 8], [2, 3],[16, 17], [13, 15],  [5, 6]):
#     # df, _ = builder.get_mean_coordination(i[0],i[1])
#     # path = r'coordination_outputs/coordination_dfs/different_densities/small field of view/coordination_df_s{}_30_small.pkl'.format(
#     #     i)
#     # coord = pickle.load(open(path, 'rb'))
#     # df, _ = builder.read_coordinationDF(coord)
#     # plt.plot(pd.DataFrame(df[:294], columns=["permute_diff"]).rolling(window=10).mean(), )
#
# plt.legend(["10k", "20k","30k", "40k",
#             "50k",  "60k", "70k"], loc='upper left',title="preliminary number of cells")
# plt.title(r'Cos of $\theta$ measure')
# plt.xticks(np.arange(0, 350, 60), labels=np.around(np.arange(0, 350, 60) * 5 / 60, decimals=1))
# # plt.yticks(np.arange(0.45, 1, 0.05))
# plt.ylim(0.6, 1, 0.5)
# plt.grid()
# plt.xlabel('Time [h]')
# plt.ylabel(r'Cos of $\theta$')
# # plt.savefig("Coordination over time different distances")
# plt.show()
# plt.close(fig)
