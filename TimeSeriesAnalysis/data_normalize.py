from abc import ABCMeta, abstractmethod, ABC

from sklearn.preprocessing import StandardScaler


class DataNormalizerStrategy(object):
    '''
    An abstract base class for defining models. The interface,
    to be implemented by subclasses, define standard model
    operations
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = name

    def preprocess_data(self, data):
        # todo add features option
        data = self.drop_columns(data)
        data = self.normalize_data(data)
        return data

    @abstractmethod
    def drop_columns(self, data, added_features=None):
        # Abstract method for transforming a full dataframe
        pass

    @abstractmethod
    def normalize_data(self, data):
        # Abstract method for transforming a full dataframe
        pass


class ActinIntensityDataNormalizer(DataNormalizerStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'actin_intensity'
        super(ActinIntensityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        to_keep = ['min', 'max', 'mean', 'sum', 'Spot track ID', 'Spot frame']
        if added_features:
            to_keep.extend(added_features)
        return data[to_keep]

    def normalize_data(self, data):
        columns = [e for e in list(data.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data


class MotilityDataNormalizer(DataNormalizerStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'motility'
        super(MotilityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        keep_features = ['Spot frame', 'Spot track ID', 'target', 'Spot position X (µm)', 'Spot position Y (µm)']
        if added_features:
            keep_features.extend(added_features)
        keep_features = [value for value in keep_features if value in data.columns]

        return data[keep_features]

    def normalize_data(self, data):
        data = data.sort_values(by=['Spot frame'])
        for label, label_df in data.groupby('Spot track ID'):
            to_reduce_x = label_df.iloc[0]['Spot position X (µm)']
            to_reduce_y = label_df.iloc[0]["Spot position Y (µm)"]

            data.loc[data['Spot track ID'] == label, 'Spot position X (µm)'] = label_df['Spot position X (µm)'].apply(
                lambda num: num - to_reduce_x)

            data.loc[data['Spot track ID'] == label, 'Spot position Y (µm)'] = label_df['Spot position Y (µm)'].apply(
                lambda num: num - to_reduce_y)

        #########################

        # if 'delta Spot position X (µm)' in df.columns:
        #     df['Spot position X (µm)'] = df['Spot position X (µm)'].diff()
        #
        # if 'delta Spot position Y (µm)' in df.columns:
        #     df['Spot position Y (µm)'] = df['Spot position Y (µm)'].diff()
        #
        # df.fillna(0, inplace=True)
        return data


class NucleiIntensityDataNormalizer(DataNormalizerStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self):
        name = 'nuclei_intensity'
        super(NucleiIntensityDataNormalizer, self).__init__(name)

    def drop_columns(self, data, added_features=None):
        to_keep = ['x', 'y', 'max_nuc', 'mean_nuc', 'min_nuc', 'nuc_size',
                   'std_nuc', 'sum_nuc', 'Spot track ID', 'Spot frame', 'target']
        if added_features:
            to_keep.extend(added_features)

    def normalize_data(self, data):
        columns = [e for e in list(data.columns) if e not in ('Spot frame', 'Spot track ID', 'target')]
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data
