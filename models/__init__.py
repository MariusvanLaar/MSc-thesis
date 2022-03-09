# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:18:38 2022

@author: Marius
"""

from .model import *

model_set = {
    "PQC-1A": PQC_1A,
    "PQC-1Y": PQC_1Y,
    "PQC-2A": PQC_1A,
    "PQC-2Y": PQC_2Y,
    "PQC-3B": PQC_3B,
    "PQC-3D": PQC_3D,
    "ANN-01": NeuralNetwork,
    }