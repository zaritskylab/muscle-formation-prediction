from CoordinationCalc import CoordinationCalc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from CoordinationGraphBuilder import CoordGraphBuilder
from matplotlib.colors import ListedColormap


class CoordinationValidations():

    def __init__(self, SMOOTHING_VAR, NEIGHBORING_DISTANCE, xml_path):
        self.coord = CoordinationCalc(SMOOTHING_VAR, NEIGHBORING_DISTANCE, xml_path)

    def validate_coord_large_quantities(self, permutations_num, file_saving_nave):
        self.coord.build_coordination_df(validation=True)
        for i in range(permutations_num):
            permutations_coefficients = []
            my_coord = self.build_coordination_df(validation=True)
            my_coefficient = self.et_coefficients(my_coord)
            permutations_coefficients.append(my_coefficient)
        df = pd.DataFrame(permutations_coefficients, columns=["coefficients"])
        df.to_csv(file_saving_nave + '.csv', index=False)

    def validate_change_distances(self, video):
        for i in (50, 100, 150, 200, 250, 300, 350, 400, 450):
            self.coord.NEIGHBORING_DISTANCE = i
            self.coord.build_coordination_df(validation=False)
            self.coord.save_coordinationDF("validation_s{}_rings_{}_mikro.pkl".format(video, i))

    def plot_validate_distances(self, name_for_saving):
        window = 25
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111)

        for i in (50, 100):
            coord_path = "coordination_outputs/validations/pickled coordination dfs/0104_rings/range_50/validation_s1_rings_{}_mikro.pkl".format(
                i)
            builder = CoordGraphBuilder(coord_path, coord_path, "", "")
            # Read coordination DFs
            control, time_c = builder.read_coordinationDF(builder.coord_control)
            # Plot
            ax1.plot(pd.DataFrame(control[:-7], columns=["permute_diff"]).rolling(window=window).mean(), )

        null_path = r"C:\Users\Amit\Desktop\Amit\ISE\3rd Year\Thesis\Analysis\Tracking\costheta\all_random_angles_video_3"
        builder = CoordGraphBuilder(null_path, null_path, "", "")
        coord, _ = builder.read_coordinationDF(builder.coord_control)
        ax1.plot(pd.DataFrame(coord[:-7], columns=["permute_diff"]).rolling(window=window).mean(), '--', c="")

        colormap = plt.cm.get_cmap('Oranges_r', 512)  # nipy_spectral, Set1,Paired
        colormap = ListedColormap(colormap(np.linspace(0, 0.65, 256)))
        colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
        for i, j in enumerate(ax1.lines):
            j.set_color(colors[i])

        plt.legend(['20-30', '60-70', '150-160', 'Randomized null model'], title="neighboring distance (0.548*10^-6 m)",
                   loc=2, fontsize='small', fancybox=True)
        plt.title(r'Coordination over time with different neighboring distances - ERK inhibitor')
        plt.xticks(np.arange(0, 921, 100), labels=np.around(np.arange(0, 921, 100) * 1.5 / 60, decimals=1))
        # plt.yticks(np.arange(0.45, 1, 0.05))
        plt.ylim(0.6, 0.88, 0.25)
        plt.grid()
        plt.xlabel('Time (hours)')
        plt.ylabel(r'Coordination')
        # plt.savefig('Coordination_over_time_distances-ERK.eps', format='eps')
        plt.show()
        plt.close(fig)

    def coordination_over_neighboring_distance(self):
        data = []
        for i in range(1, 10):
            coord_path = "coordination_outputs/validation/validation_s3_0.{}.pkl".format(i)
            builder = CoordGraphBuilder(coord_path, coord_path, "", "")
            # Read coordination DFs
            control, time_c = builder.read_coordinationDF(builder.coord_control)
            data.append([control[0], i / 10])

        df = pd.DataFrame(data, columns=['cos_theta', 'neighboring_distance'])
        fig = plt.figure(figsize=(6, 4))
        plt.plot(df["neighboring_distance"], df["cos_theta"])
        plt.legend()
        plt.title(r'Cos of $\theta$ over neighboring distance')
        plt.xlabel('neighboring distance')
        plt.ylabel(r'Cos of $\theta$')
        # plt.savefig(name_for_saving)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    for i in range(1,13):
        validator = CoordinationValidations(SMOOTHING_VAR=5, NEIGHBORING_DISTANCE=0.3,
                                        xml_path=r"muscle-formation-diff/data/tracks_xml/0104/Experiment1_w1Widefield550_s{}_all_0104.xml".format(i))
        # validator.validate_change_distances(video=i)
    validator.plot_validate_distances("neighboring distance changes- control")
