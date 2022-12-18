import glob
import pickle
import os, sys

sys.path.append('/sise/home/reutme/muscle-formation-regeneration')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from TimeSeriesAnalysis.data_preprocessing.merge_preprocessed_data_chunks import concat_data_portions
import TimeSeriesAnalysis.params as params
import TimeSeriesAnalysis.consts as consts



if __name__ == '__main__':
    print("running merge_prepare_data_chunks")

    # get arguments from sbatch
    modality = sys.argv[1]

    s_run = consts.s_runs[os.getenv('SLURM_ARRAY_TASK_ID')]

    feature_type = sys.argv[2]

    for win_size in params.feature_calc_types[feature_type]:

        concat_data_portions(params.local_density, win_size, s_run, modality, feature_type)







