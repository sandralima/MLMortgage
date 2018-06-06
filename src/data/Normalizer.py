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
        Q1 = data.apply(lambda x: x.quantile(self.rangeiq[0]), axis=1)
        Q3 = data.apply(lambda x: x.quantile(self.rangeiq[1]), axis=1)
        self.scale_ = Q3 - Q1
        self.center_ = Q1        
    
    
    def fit_minmax (self, data):
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