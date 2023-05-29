from data_layer.data_preprocessing.features_calculator import *
from data_layer.data_preprocessing.data_normalizer import *
from tsfresh.utilities.dataframe_functions import impute_dataframe_zero, impute


class DataPreprocessor:
    """ Holds all data preprocessing operations """

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


def data_preprocessor_factory(modality, transformer_name="time_split_transform", impute_func="impute_zeroes",
                              impute_methodology="ImputeAllData"):
    """
    Factory Method for creating a specific data preprocessor, according to the needed modality/operations. :param
    modality: (Str), can be "motility", "actin_intensity", "nuclei_intensity" or "local_density". :param
    transformer_name: (Str), determines the chosen tsfresh transformation method. can be "time_split_transform" or
    "single_cell_transform". Default transformer is "time_split_transform". :param impute_func: (Str), determines the
    chosen imputation function. can be "impute" or "impute_zeroes", default is "impute_zeroes". :param
    impute_methodology: (Str), determines the chosen imputation method. can be "ImputeSingleCell", "ImputeAllData" or
    "ImputeTimeSlots". default imputation method is "ImputeAllData". :return:
    """
    data_operators = {
        "local_density": (LocalDensityFeaturesCalculator, LocalDensityDataNormalizer),
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
