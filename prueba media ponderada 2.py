import numpy as np
def weighted_median(arrays, weights):
    weighted_medians = []
    for i in range(arrays[0].shape[0]):
        for j in range(arrays[0].shape[1]):
            values = [array[i][j] for array in arrays]
            indices = np.argsort(values)
            values = np.array(values)[indices]
            weights = np.array(weights)[indices]
            cumulative_weights = np.cumsum(weights)
            median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
            weighted_medians.append(values[median_index])
    return np.array(weighted_medians).reshape(arrays[0].shape)

def weighted_median_1d(arrays, weights):
    weighted_medians = []
    for i in range(len(arrays[0])):
        values = [array[i] for array in arrays]
        indices = np.argsort(values)
        values = np.array(values)[indices]
        weights = np.array(weights)[indices]
        #cumulative_weights = np.cumsum(weights)[indices]
        cumulative_weights = np.cumsum(weights)
        median_index = np.searchsorted(cumulative_weights, 0.5 * cumulative_weights[-1])
        weighted_medians.append(values[median_index])
    return np.array(weighted_medians)

arr2 = np.array([[-300, 2, 3], [4, 5, 6], [7, 8, 9]])
arr1 = np.array([[-0.5, 20, 30], [40, 50, 60], [70, 80, 90]])
arr4 = np.array([[0.25, 20, 30], [40, 50, 60], [70, 80, 90]])
arr5 = np.array([[0.48, 20, 30], [40, 50, 60], [70, 80, 90]])
arr6 = np.array([[-0.25, 20, 30], [40, 50, 60], [70, 80, 90]])
arr3 = np.array([[100000, 200, 300], [400, 500, 600], [700, 800, 900]])



#print(np.average([arr1, arr2, arr3],axis=0,weights=[0.4,0.4,0.2]))


arr1 = np.array([100, 2, 3, 4, 5])
arr2 = np.array([0.25, -7, 8, 9, 10])
arr3 = np.array([-0.5, 12, 13, 14, 15])
arr4 = np.array([-0.8, 10, 13, 14, 15])
arr5 = np.array([0.7, 40, 13, 14, 15])
arr6 = np.array([0.1, 1, 13, 14, 15])
arr7 = np.array([0.3, 12, 13, 14, 15])

try:
    print(weighted_median([arr1, arr2, arr3, arr4, arr5, arr6], [1, 1, 1,1,1,1]))
except IndexError:

    print(weighted_median_1d([arr1, arr2, arr3, arr4, arr5, arr6, arr7], [1, 1, 1, 1, 1, 1, 1]))
