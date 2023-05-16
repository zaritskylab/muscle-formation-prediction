import pandas as pd
import pickle


def convert_score_df(score_df, modality):
    """converts the scores dataframe from horizontal to vertical view"""
    df = pd.DataFrame()
    for i in range(len(score_df)):
        track = score_df.iloc[i, :]
        tmp_df = pd.DataFrame({f"score_{modality}": track.drop(index="Spot track ID")})
        tmp_df["time"] = tmp_df.index * 5 / 60
        tmp_df["Spot frame"] = tmp_df.index
        tmp_df["Spot track ID"] = int(track["Spot track ID"])
        df = df.append(tmp_df, ignore_index=True)
    return df


def get_scores_df(scores_motility_path, scores_intensity_path):
    """
    The method receives paths of motility & actin intensity differentiation scores, rotates them vertically and merges them into one dataframe.
    :param scores_motility_path: (Str) paths of motility differentiation scores' dataframe
    :param scores_intensity_path: (Str) paths of actin intensity differentiation scores' dataframe
    :return: (pd.DataFrame) differntiation scores by motility & actin intensity models
    """
    scores_df_mot = convert_score_df(pickle.load(open(scores_motility_path, 'rb')), "motility")
    scores_df_int = convert_score_df(pickle.load(open(scores_intensity_path, 'rb')), "intensity")
    scores_df = pd.merge(left=scores_df_mot, right=scores_df_int, on=["Spot track ID", "Spot frame", "time"])

    return scores_df
