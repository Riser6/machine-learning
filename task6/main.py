import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

mat = loadmat('./data/ex6data1.mat')
print(mat.keys())