# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:03:47 2022

@author: Marius
"""

from .dataclasses import *

#all_datasets = [
#    "CIFAR-15",
#    "CIFAR-PCA-15", #PCA on all 3072 pixels for classes 1 and 5
#    "Havlicek-10", #N-features is 10
#    "Havlicek-easy-10", #N-features is 10
#    "Simple-10", #N-features is 10
#    "wdbc", # Breast cancer, 30 features
#    "CIFAR-PCA-08", #PCA on all 3072 pixels for classes 0 and 8
#    ]
    
all_datasets = {
    "simple": SIMPLE,
    "wdbc": WDBC,
    "synth-4A": SYNTH_4A,
    "synth-4AA": SYNTH_4AA,
    "ion": ION, 
    "sonar": SONAR,
    "spectf": SPECTF,
    "mnist": MNIST,
    }