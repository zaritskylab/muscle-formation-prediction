from data_preprocessing.features_calculator import *
from data_preprocessing.data_normalizer import *
from data_preprocessing.imputer_strategy import *
from data_preprocessing.tsfresh_transformer import *
from tsfresh.utilities.dataframe_functions import impute_dataframe_zero, impute


class DataPreprocessor:
    """ Object that holds all data preprocessing operations """

    def __init__(self, features_creator, data_normalizer, tsfresh_transformer, imputer):
        self.feature_creator = features_creator()
        self.data_normalizer = data_normalizer()
        self.tsfresh_transformer = tsfresh_transformer()
        self.imputer = imputer

    def __str__(self) -> str:
        my_str = self.feature_creator.name \
                 + " " + self.data_normalizer.name \
                 + " " + self.tsfresh_transformer.name \
                 + " " + self.imputer.name
        return my_str


def data_preprocessor_factory(modality, transformer_name, impute_func, impute_methodology):
    """Factory Method"""
    data_operators = {
        "motility": (MotilityFeaturesCalculator, MotilityDataNormalizer),
        "actin_intensity": (ActinIntensityFeaturesCalculator, ActinIntensityDataNormalizer),
        "nuclei_intensity": (NucleiIntensityFeaturesCalculator, NucleiIntensityDataNormalizer),
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
        "ImputeSingleCell": ImputerSingleCell(imputing_method),
        "ImputeAllData": ImputerAllData(imputing_method),
        "ImputeTimeSlots": ImputerTimeSlots(imputing_method),
    }

    (features_creator, data_normalizer) = data_operators[modality]
    transformer = tsfresh_transformers[transformer_name]
    imputer = imputers[impute_methodology]

    preprocessor = DataPreprocessor(features_creator, data_normalizer, transformer, imputer)

    return preprocessor
