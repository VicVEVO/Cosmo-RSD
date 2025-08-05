from numba import njit

@njit
def integral_trapezoid(func, a, b, N, H0, omega_m):
    h = (b - a) / N
    result = 0.5 * (func(a, H0, omega_m) + func(b, H0, omega_m))
    for i in range(1, N):
        result += func(a + i * h, H0, omega_m)
    result *= h
    return result

@njit
def find_index(x, x_array, delta_x):
    """Find corresponding index for x in an x_array array.

    Args:
        x (float): the x value we have
            Must be between x_array[0] and x_array[-1]
        x_array (numpy array): the array with all values of x
            Must be linspace and sorted
        delta_x (float): the precision of the linspace array

    Returns:
        int: the corresponding index
    """
    return int((x - x_array[0]) / delta_x + 0.5)