import shap
import TimeSeriesAnalysis.diff_tracker_utils as utils
import pandas as pd

from TimeSeriesAnalysis.intensity_erk_compare import get_tracks_list

window = 30
con_windows = [[0, 30], [40, 70], [90, 120], [140, 170], [180, 210], [220, 250]]
diff_window = [140, 170]

second_dir = f"{diff_window} frames ERK, {con_windows} frames con"
dir_path = f"18-02-2022-manual_mastodon_motility" + "/" + second_dir
clf, X_train, X_test, y_train, y_test = utils.load_data(dir_path)
explainer = shap.TreeExplainer(clf)

csv_path = fr"../data/mastodon/test/Nuclei_2-vertices.csv"
df_s2 = pd.read_csv(csv_path, encoding="ISO-8859-1")
df_s2 = df_s2[df_s2["manual"] == 1]
tracks_s2 = get_tracks_list(df_s2)

csv_path = fr"../data/mastodon/test/Nuclei_3-vertices.csv"
df_s3 = pd.read_csv(csv_path, encoding="ISO-8859-1")
df_s3 = df_s3[df_s3["manual"] == 1]
tracks_s3 = get_tracks_list(df_s3)

df_total_feature_importance = pd.DataFrame()
for track in tracks_s2:
    if len(track > 150):
        track = track.sort_values("Spot frame")
        true_p = {}
        for i in range(0, len(track), 1):
            if i + window > len(track):
                break
            track_portion = track[i:i + window]
            max_frame = track_portion["Spot frame"].max()
            X = utils.extract_distinct_features(df=track_portion, feature_list=X_train.columns)
            probs = clf.predict_proba(X)
            true_p[max_frame] = probs[0][1]
            shap_values = explainer.shap_values(X)
            feature_importance = pd.DataFrame(list(zip(X_train.columns, shap_values)),
                                              columns=['col_name', 'feature_importance_vals'])
            feature_importance["track_portion_end_t"] = i + window
            feature_importance["true_prob"] = probs[0][1]
            feature_importance["target"] = 0
            df_total_feature_importance = df_total_feature_importance.append(feature_importance, ignore_index=True)

            print(f"track portion: [{i}:{i + window}]")
            print(clf.classes_)
            print(probs)

