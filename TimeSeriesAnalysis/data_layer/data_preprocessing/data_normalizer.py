from abc import ABCMeta, abstractmethod, ABC
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataNormalizerStrategy(object):
    """
    An abstract base class for data normalization. The interface,
    to be implemented by subclasses, define standard data normalizing operations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name  # normalizer's name

    def preprocess_data(self, data) -> pd.DataFrame:
        """
        The method activates data normalization process.
        It removes irrelevant columns and performs data normalization.
        :param data: (pd.DataFrame) single cells trajectories
        :return: (pd.DataFrame) single cells normalized trajectories
        """
        print("normalize data", flush=True)
        if not data.empty:
            data = self.drop_columns(data)
            data = self.normalize_data(data)

        return data

    @abstractmethod
    def drop_columns(self, data, added_features=None) -> pd.DataFrame:
        """
        Abstract method for removing irrelevant columns from the trajectories' DataFrame.
        :param data: (pd.DataFrame) single cells trajectories
        :param added_features: (List) columns of features names to keep in the DataFrame.
        :return: (pd.DataFrame) single cells trajectories with needed columns only.
        """
        pass

    @abstractmethod
    def normalize_data(self, data) -> pd.DataFrame:
        """
        Abstract method for normalising a full DataFrame.
        :param data: (pd.DataFrame) single cells trajectories
        :return: (pd.DataFrame) single cells normalized trajectories.
        """
        pass


class ActinIntensityDataNormalizer(DataNormalizerStrategy, ABC):
    """
    Data normalizer for transforming single cell trajectories with actin intensity measurements.
    """

    def __init__(self):
        name = 'actin_intensity'
        super(ActinIntensityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        """
        Removes irrelevant columns from the trajectories' DataFrame, keeps only actin intensity features.
        :param data: (pd.DataFrame) single cells trajectories
        :param added_features: (List) columns of features names to keep in the DataFrame.
        :return: (pd.DataFrame) single cells trajectories with needed columns only.
        """
        to_keep = ['min', 'max', 'mean', 'sum', 'Spot track ID', 'Spot frame']
        if added_features:
            to_keep.extend(added_features)

        return data[to_keep]

    def normalize_data(self, data):
        """
        Normalises a full DataFrame of actin intensity measurements.
        We use scikit-learn's StandardScaler, which standardize features by removing
        the mean and scaling to unit variance.
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        :param data: (pd.DataFrame) single cells trajectories
        :return: (pd.DataFrame) single cells normalized trajectories.
        """
        columns = [e for e in list(data.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])

        return data


class MotilityDataNormalizer(DataNormalizerStrategy, ABC):
    """
    Data normalizer for transforming single cell trajectories with motility measurements.
    """

    def __init__(self):
        name = 'motility'
        super(MotilityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        """
        Removes irrelevant columns from the trajectories' DataFrame, keeps only motility features.
        :param data: (pd.DataFrame) single cells trajectories
        :param added_features: (List) columns of features names to keep in the DataFrame.
        :return: (pd.DataFrame) single cells trajectories with needed columns only.
        """
        keep_features = ['Spot frame', 'Spot track ID', 'target', 'Spot position X', 'Spot position Y']
        if added_features:
            keep_features.extend(added_features)
        keep_features = [value for value in keep_features if value in data.columns]

        return data[keep_features]

    def normalize_data(self, data):
        """
        Normalises a full DataFrame of motility measurements.
        We calculate the displacement of a single cell for each time point.
        :param data: (pd.DataFrame) single cells trajectories
        :return: (pd.DataFrame) single cells normalized trajectories.
        """
        data = data.sort_values(by=['Spot frame'])
        for label, label_df in data.groupby('Spot track ID'):
            to_reduce_x = label_df.iloc[0]['Spot position X']
            to_reduce_y = label_df.iloc[0]["Spot position Y"]

            data.loc[data['Spot track ID'] == label, 'Spot position X'] = label_df['Spot position X'].apply(
                lambda num: num - to_reduce_x)

            data.loc[data['Spot track ID'] == label, 'Spot position Y'] = label_df['Spot position Y'].apply(
                lambda num: num - to_reduce_y)

        return data


class NucleiIntensityDataNormalizer(DataNormalizerStrategy, ABC):
    """
    Data normalizer for transforming single cell trajectories with nuclei intensity measurements.
    """

    def __init__(self):
        name = 'nuclei_intensity'
        super(NucleiIntensityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        """
        Removes irrelevant columns from the trajectories' DataFrame, keeps only nuclei intensity features.
        :param data: (pd.DataFrame) single cells trajectories
        :param added_features: (List) columns of features names to keep in the DataFrame.
        :return: (pd.DataFrame) single cells trajectories with needed columns only.
        """
        to_keep = ['aspect_ratio', 'nuc_size', 'Spot track ID', 'Spot frame']
        if added_features:
            to_keep.extend(added_features)

        return data[to_keep]

    def normalize_data(self, data):
        """
        Normalises a full DataFrame of nuclei intensity measurements.
        We use scikit-learn's StandardScaler, which standardize features by removing
        the mean and scaling to unit variance.
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

        :param data: (pd.DataFrame) single cells trajectories
        :return: (pd.DataFrame) single cells normalized trajectories.
        """
        columns = [e for e in list(data.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])

        return data


class LocalDensityDataNormalizer(DataNormalizerStrategy, ABC):
    """
    Data normalizer for transforming single cell trajectories with local density measurement.
    """

    def __init__(self):
        name = 'local_density'
        super(LocalDensityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        """
        Removes irrelevant columns from the trajectories' DataFrame, keeps only local density feature.
        :param data: (pd.DataFrame) single cells trajectories
        :param added_features: (List) columns of features names to keep in the DataFrame.
        :return: (pd.DataFrame) single cells trajectories with needed columns only.
        """
        to_keep = ['local density', 'Spot track ID', 'Spot frame']
        if added_features:
            to_keep.extend(added_features)

        return data[to_keep]

    def normalize_data(self, data):
        """
        No normalization is performed.
        :param data: (pd.DataFrame) single cells trajectories.
        :return: (pd.DataFrame) single cells trajectories.
        """
        return data
