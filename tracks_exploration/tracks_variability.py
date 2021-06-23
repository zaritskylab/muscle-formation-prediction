import pandas as pd
from DataPreprocessing.load_tracks_xml import *
from numpy.random import seed
from scipy.stats import f_oneway
# import sweetviz as sv
from pandas_profiling import ProfileReport

# importing Autoviz class
# from autoviz.AutoViz_Class import AutoViz_Class  # Instantiate the AutoViz class

# seed the random number generator
from ts_fresh import concat_dfs, normalize_tracks, drop_columns

seed(1)


def get_df(ind):
    xml_path = r"data/tracks_xml/0104/Experiment1_w1Widefield550_s{}_all_0104.xml".format(ind)
    _, df = load_tracks_xml(xml_path)
    return df


def get_exp_statistics():
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
        xml_path = r"data/tracks_xml/0104/Experiment1_w1Widefield550_s{}_all_0104.xml".format(i)
        _, df = load_tracks_xml(xml_path)
        stats_df = pd.DataFrame(df.describe())
        stats_df.to_excel("stats_df_{}.xls".format(i))


def anova(*args):
    print("Fail to Reject H0: Paired sample distributions are equal.\n"
          "Reject H0: Paired sample distributions are not equal")
    for col in args[0].columns:
        to_compare = [arg[col] for arg in args]
        # compare samples
        stat, p = f_oneway(*to_compare)

        print("Analysis of Variance Test for column {}".format(col))
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print('Same distributions (fail to reject H0)')
        else:
            print('Different distributions (reject H0)')


# df1 = get_df(1)
# df2 = get_df(2)
# df3 = get_df(3)
# df4 = get_df(4)
# df5 = get_df(5)
# df6 = get_df(6)
# df7 = get_df(7)
# df8 = get_df(8)
# df9 = get_df(9)
# df10 = get_df(10)
# df11 = get_df(11)
# df12 = get_df(12)

lst_videos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 1, 2, 3, 4
df = concat_dfs(lst_videos, motility=False, intensity=True)
df = drop_columns(df, motility=True, intensity=True)
print(df.columns)
df = normalize_tracks(df, motility=True, intensity=True)

prof = ProfileReport(df)
prof.to_file(output_file='output_normalized.html')

# df1 = sv.compare(df1, df3)
# df1.show_html('Compare.html')
