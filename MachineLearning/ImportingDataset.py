# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Matrix of features
X = dataset.iloc[:, :-1].values

# Create dependent variable martix

Y = dataset.iloc[:,3].values
