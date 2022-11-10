from abc import ABCMeta, abstractmethod, ABC

from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import sys, os
sys.path.append('/sise/home/shakarch/muscle-formation-diff')
sys.path.append(os.path.abspath('../..'))
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.diff_tracker_utils import *
import consts

class ImputerStrategy(object):
    '''
    An abstract base class for defining models. The interface,
    to be implemented by subclasses, define standard model
    operations
    '''
    __metaclass__ = ABCMeta

    def __init__(self, name, impute_func):
        self.apply_impute = impute_func
        self.name = name

    @abstractmethod
    def impute(self, data):
        # Abstract method for transforming a full dataframe
        pass


class ImputerAllData(ImputerStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self, impute_func=impute):
        name = 'ImputeAllData'
        super(ImputerAllData, self).__init__(name, impute_func)

    def impute(self, data):
        imputed = self.apply_impute(data)
        return imputed


class ImputerSingleCell(ImputerStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self, impute_func=impute):
        name = 'ImputeSingleCell'
        super(ImputerSingleCell, self).__init__(name, impute_func)

    def impute(self, data):
        imputed = pd.DataFrame()
        for label, labeled_df in data.groupby("Spot track ID"):
            imputed = imputed.append(self.apply_impute(labeled_df), ignore_index=True)

        return imputed


class ImputerTimeSlots(ImputerStrategy, ABC):
    '''
    An ordinary least squares (OLS) linear regression model
    '''

    def __init__(self, impute_func=impute):
        name = 'ImputeTimeSlots'
        super(ImputerTimeSlots, self).__init__(name, impute_func)

    def impute(self, data):
        imputed = pd.DataFrame()
        df_time_window_split_list = split_data_to_time_portions(data, consts.tracks_len)
        for t_df in df_time_window_split_list:
            imputed = imputed.append(self.apply_impute(t_df), ignore_index=True)

        return imputed
