import numpy as np

def rasterize(func, x_vals, *args):
    y_vals = np.zeros(len(x_vals))
    for i in range(0, len(x_vals)):
        y_vals[i] = func(x_vals[i], *args)
    return [x_vals, y_vals]

def complete(vals_array, left_vals, right_vals):
    for i in range(0, len(vals_array)):
        if len(left_vals) != 0:
            vals_array[i] = np.hstack((left_vals[i], vals_array[i]))
        if len(right_vals) != 0:
            vals_array[i] = np.hstack((vals_array[i], right_vals[i]))
    return vals_array
