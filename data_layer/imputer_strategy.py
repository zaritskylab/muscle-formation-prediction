from abc import ABCMeta, abstractmethod, ABC
import sys, os

sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model_layer.utils import *
from configuration import params


class ImputerStrategy(object):
    """
    An abstract base class for Imputation.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, impute_func):
        self.apply_impute = impute_func
        self.name = name

    @abstractmethod
    def impute(self, data):
        """ Abstract method for imputing a full dataframe"""
        pass


class ImputerAllData(ImputerStrategy, ABC):
    """
    A class for imputation all data as a one piece.
    """

    def __init__(self, impute_func=impute):
        name = 'ImputeAllData'
        super(ImputerAllData, self).__init__(name, impute_func)

    def impute(self, data):
        """
        imputes all data in one piece.
        :param data: (pd.DataFrame) single cells trajectories.
        :return: (pd.DataFrame) imputed single cells trajectories.
        """
        imputed = self.apply_impute(data)
        return imputed


class ImputerSingleCell(ImputerStrategy, ABC):
    """
    A class for imputation of single cell's data at a time.
    """

    def __init__(self, impute_func=impute):
        name = 'ImputeSingleCell'
        super(ImputerSingleCell, self).__init__(name, impute_func)

    def impute(self, data):
        """
        imputes single cell's data at a time.
        :param data: (pd.DataFrame) single cells trajectories.
        :return: (pd.DataFrame) imputed single cells trajectories.
        """
        imputed = pd.DataFrame()
        for label, labeled_df in data.groupby("Spot track ID"):
            imputed = imputed.append(self.apply_impute(labeled_df), ignore_index=True)

        return imputed


class ImputerTimeSlots(ImputerStrategy, ABC):
    """
    A class for imputation of the data, divided into time portions.
    """

    def __init__(self, impute_func=impute):
        name = 'ImputeTimeSlots'
        super(ImputerTimeSlots, self).__init__(name, impute_func)

    def impute(self, data):
        """
        imputes single cell's data at a time.
        :param data: (pd.DataFrame) single cells trajectories.
        :return: (pd.DataFrame) imputed single cells trajectories.
        """
        imputed = pd.DataFrame()
        df_time_window_split_list = split_data_to_time_portions(data, params.tracks_len)
        for t_df in df_time_window_split_list:
            imputed = imputed.append(self.apply_impute(t_df), ignore_index=True)

        return imputed
