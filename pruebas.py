import numpy as np
from scipy.stats import skew
import itertools
from random import sample
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
def skeweness_2d(arrays):  # el bueno
    sk_array = []
    for i in range(arrays[0].shape[0]):
        for j in range(arrays[0].shape[1]):
            values = [array[i][j] for array in arrays]
            indices = np.argsort(values)
            values = np.array(values)[indices]
            sk = np.abs(skew(values))
            sk_array.append(sk)
    return np.array(sk_array).reshape(arrays[0].shape)

def skeweness_1d(arrays):  # el bueno
    sk_array = []
    for i in range(arrays[0].shape[0]):
        values = [array[i] for array in arrays]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        sk = np.abs(skew(values))
        sk_array.append(sk)
    return np.array(sk_array).reshape(arrays[0].shape)

def skeweness(arrays):
    try:
        return skeweness_2d(arrays)
    except IndexError:
        return skeweness_1d(arrays)

arr1 = np.array([[1, 20, 0.5], [4, 5, 6], [7, 8, 9]])
arr2 = np.array([[2, 2, 0.24], [40, 50, 60], [70, 80, 90]])
arr3 = np.array([[3, 200, 0.7], [40, 50, 60], [70, 80, 90]])

arr1 = [-0.26906955, -0.17233811, -0.16652365, -0.16040955, -0.1393857,  -0.13236828,
       -0.11403103, -0.11280964, -0.10864646, -0.09910333, -0.09651558, -0.08113343,
       -0.07566889, -0.07442398, -0.06836082, -0.06595942, -0.06270833, -0.05974881,
       -0.05078789, -0.04613058, -0.03723432, -0.02926191, -0.02474026, -0.02285848,
       -0.01221434, -0.00436003, -0.00365365,  0.00256299,  0.00332146,  0.00881493,
        0.01751379,  0.02125426,  0.02881968,  0.03557607,  0.03851334,  0.0391941,
        0.03944889,  0.04504812,  0.06456483,  0.07145527,  0.08000311,  0.0911712,
        0.10376644,  0.11895037,  0.12060568,  0.18284555,  0.18841994,  0.20366146,
        0.21319088,  0.2137932]

arr2 = [-0.70797193, -0.32199258, -0.25426877 ,-0.24929462 ,-0.15009546, -0.13730797,
 -0.13454093, -0.12883283, -0.12388984, -0.11356864, -0.10777549, -0.10439844,
 -0.0990753 , -0.09699489, -0.09597599, -0.08948115, -0.08185728, -0.07967483,
 -0.06505378, -0.06346685, -0.06035731, -0.04455213, -0.04030947, -0.00998174,
  0.00800515,  0.02554614,  0.05948052,  0.06630304,  0.07256768,  0.07300469,
  0.07535718,  0.07942355,  0.08580377,  0.08808801,  0.09802654,  0.10032529,
  0.10181952,  0.10386834,  0.15808704,  0.16067746,  0.18264684,  0.22574724,
  0.27390388,  0.28118634,  0.2923232 ,  0.3364188,   0.4061594,   0.41759333,
  0.48158437,  0.48385382]


f = Fitter(arr1,
           distributions= get_common_distributions())
f.fit()
print(np.asarray(f.summary()))
print(f.get_best(method = 'sumsquare_error'))

f = Fitter(arr2,
           distributions= get_common_distributions())
f.fit()
print(f.summary())
print(f.get_best(method = 'sumsquare_error'))

#print(skeweness([arr1,arr2,arr3]))
