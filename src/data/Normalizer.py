# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:42:14 2018

@author: sandr
"""

import numpy as np


class Normalizer(object):
    
    def __init__(self, scales=None, centers=None, rangeiq=[0.25, 0.75]):
        self.scale_ = scales
        self.center_ = centers
        self.rangeiq = rangeiq        
        
        
    def fit_quantile (self, data):
        '''
        RobustScaler uses a similar method to the Min-Max scaler. However, it uses the interquartile range instead of the min-max, 
        which makes it robust to outliers.  For each feature: (xi−Q1(x))/IQR, where IQR = (Q3(x)−Q1(x)). 
        Following the gold rule we can choose Q1=0.25 and Q3=0.75. 
        According this, Current_Interest_Rate, for example, has 9.131.472 outliers approximately.
        Args: 
            data: data which will be apply the scale and center.
        Returns: 
            None. Normalizer Object initialized.
        Raises:
        '''
        
        Q1 = data.apply(lambda x: x.quantile(self.rangeiq[0]), axis=1)
        Q3 = data.apply(lambda x: x.quantile(self.rangeiq[1]), axis=1)
        self.scale_ = Q3 - Q1
        self.center_ = Q1        
    
    
    def fit_minmax (self, data):
        '''
        The MinMax Scaler (xi−min(x))/(max(x)−min(x)) shrinks the range such that it is now between 0 and 1 (or -1 to 1 if there exist negative values). 
        The MinMaxScaler works well for cases when the distribution is not Gaussian or when the standard deviation is very small. 
        However, it is sensitive to outliers.
        Args: 
            data: data which will be apply the scale and center.
        Returns: 
            None. Normalizer Object initialized.
        Raises:
        '''
        MIN = data.apply(lambda x: x.min(), axis=1)
        MAX = data.apply(lambda x: x.max(), axis=1)
        self.scale_ = MAX - MIN
        self.center_ = MIN    
    
    
    def scale(self, x, i):
        if self.scale_[i] !=0:
            return (x - self.center_[i]) / (self.scale_[i])
        else: return np.nan
    
    
    def transform(self, data):        
        datai = data.apply(lambda x: x.map(lambda z: self.scale(z, data.columns.get_loc(x.name))))
        return datai