import piv_functions as piv
import numpy as np

test_array = np.array([[3, 1, np.nan, 5, np.nan],
                       [2, 4, 6, np.nan, 8],
                       [np.nan, 9, 10, 11, 12]])

print(-np.sort(-test_array, axis=-2))