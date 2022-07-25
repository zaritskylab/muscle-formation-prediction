from calc_features import *
from data_normalize import *
from impute_strategy import *
from tsfresh_transformer import *
from tsfresh.utilities.dataframe_functions import impute_dataframe_zero, impute


class DataPreprocessor:
    """ it simply returns the french version """

    def __init__(self, features_creator, data_normalizer, tsfresh_transformer, imputer):
        self.feature_creator = features_creator()
        self.data_normalizer = data_normalizer()
        self.tsfresh_transformer = tsfresh_transformer()
        self.imputer = imputer


def data_preprocessor_factory(modality, transformer_name, impute_func, impute_methodology):
    """Factory Method"""
    data_operators = {
        "motility": (MotilityCalcFeatures, MotilityDataNormalizer),
        "actin_intensity": (ActinIntensityCalcFeatures, ActinIntensityDataNormalizer),
        "nuclei_intensity": (NucleiIntensityCalcFeatures, NucleiIntensityDataNormalizer),
    }
    tsfresh_transformers = {
        "time_split_transform": TimeSplitsTSFreshTransform,
        "single_cell_transform": SingleCellTSFreshTransform,
    }
    imputing_functions = {
        "impute": impute,
        "impute_zeroes": impute_dataframe_zero
    }
    imputing_method = imputing_functions[impute_func]
    imputers = {
        "ImputeSingleCell": ImputeSingleCell(imputing_method),
        "ImputeAllData": ImputeAllData(imputing_method),
        "ImputeTimeSlots": ImputeTimeSlots(imputing_method),
    }

    (features_creator, data_normalizer) = data_operators[modality]
    transformer = tsfresh_transformers[transformer_name]
    imputer = imputers[impute_methodology]

    preprocessor = DataPreprocessor(features_creator, data_normalizer, transformer, imputer)

    return preprocessor
