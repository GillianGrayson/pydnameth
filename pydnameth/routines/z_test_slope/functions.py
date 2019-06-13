import numpy as np
from scipy.stats import norm


def z_test_slope_proc(slopes, slopes_std, num_subjects, metrics_dict):

    std_errors = [slopes_std[i] / np.sqrt(num_subjects[i]) for i in range(0, len(slopes_std))]
    z_value = (slopes[0] - slopes[1]) / np.sqrt(sum([std_error * std_error for std_error in std_errors]))
    p_value = norm.sf(abs(z_value)) * 2.0

    metrics_dict['z_value'].append(z_value)
    metrics_dict['p_value'].append(p_value)
    metrics_dict['abs_z_value'].append(np.absolute(z_value))
