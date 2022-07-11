# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:03:47 2022

@author: Marius
"""

from .dataclasses import *
    
all_datasets = {
    "simple": SIMPLE,
    "wdbc": WDBC,
    "synth-4A": SYNTH_4A,
    "synth-4F": SYNTH_4F,
    "synth-4F-rand": SYNTH_4F_random,
    "ion": ION, 
    "sonar": SONAR,
    "spectf": SPECTF,
    "mnist-13": MNIST_13,
    "mnist-23": MNIST_23,
    "ising-10": ISING_10,
    "circle": CIRCLE,
    }