import numpy as np
import statsmodels.api as sm


def get_box_xs(targets):
    targets = np.squeeze(np.asarray(targets))
    min_target = int(np.floor(min(targets)))
    max_target = int(np.ceil(max(targets)))
    xs = list(range(min_target, max_target + 1))
    return xs


def process_box(targets, values, semi_window=2, box_b='left', box_t='right'):
    targets = np.squeeze(np.asarray(targets))
    values = np.squeeze(np.asarray(values))

    min_target = int(np.floor(min(targets)))
    max_target = int(np.ceil(max(targets)))
    window_residuals = {}
    window_targets = {}
    for center in range(min_target, max_target + 1):
        window_residuals[center] = []
        window_targets[center] = []

    for point_id in range(0, len(targets)):
        curr_target = targets[point_id]
        curr_target_round = int(round(curr_target))
        curr_residuals = values[point_id]
        for window_id in range(curr_target_round - semi_window, curr_target_round + semi_window + 1):
            if window_id in window_residuals:
                window_residuals[window_id].append(curr_residuals)
                window_targets[window_id].append(curr_target)

    xs = list(window_targets.keys())
    bs = np.zeros(len(xs), dtype=float)
    ms = np.zeros(len(xs), dtype=float)
    ts = np.zeros(len(xs), dtype=float)
    for x_id in range(0, len(xs)):
        curr_residuals = window_residuals[xs[x_id]]
        q5, q1, median, q3, q95 = np.percentile(np.asarray(curr_residuals), [5, 25, 50, 75, 95])
        iqr = q3 - q1
        ms[x_id] = median
        if box_b == 'left':
            bs[x_id] = q1 - 1.5 * iqr
        elif box_b == 'Q1':
            bs[x_id] = q1
        elif box_b == 'Q5':
            bs[x_id] = q5
        else:
            raise ValueError('Unknown box_b type')

        if box_t == 'right':
            ts[x_id] = q3 + 1.5 * iqr
        elif box_t == 'Q3':
            ts[x_id] = q3
        elif box_t == 'Q95':
            ts[x_id] = q95
        else:
            raise ValueError('Unknown box_t type')

    return xs, bs, ms, ts


def process_line_fit(exog, endog, characteristics_dict, key_prefix, suffix):
    is_same_elements = all(x == endog[0] for x in endog)
    if is_same_elements:
        characteristics_dict[key_prefix + '_lin_lin_R2' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_lin_intercept' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_lin_slope' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_lin_intercept_std' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_lin_slope_std' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_lin_intercept_p_value' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_lin_slope_p_value' + suffix].append('NA')
        R2s = [-1]
    else:
        lin_lin_exog = sm.add_constant(exog)
        lin_lin_endog = endog
        lin_lin_results = sm.OLS(lin_lin_endog, lin_lin_exog).fit()
        characteristics_dict[key_prefix + '_lin_lin_R2' + suffix].append(lin_lin_results.rsquared)
        characteristics_dict[key_prefix + '_lin_lin_intercept' + suffix].append(lin_lin_results.params[0])
        characteristics_dict[key_prefix + '_lin_lin_slope' + suffix].append(lin_lin_results.params[1])
        characteristics_dict[key_prefix + '_lin_lin_intercept_std' + suffix].append(lin_lin_results.bse[0])
        characteristics_dict[key_prefix + '_lin_lin_slope_std' + suffix].append(lin_lin_results.bse[1])
        characteristics_dict[key_prefix + '_lin_lin_intercept_p_value' + suffix].append(lin_lin_results.pvalues[0])
        characteristics_dict[key_prefix + '_lin_lin_slope_p_value' + suffix].append(lin_lin_results.pvalues[1])
        R2s = [lin_lin_results.rsquared]

    if min(endog) > 0:
        lin_log_exog = sm.add_constant(exog)
        lin_log_endog = np.log(endog)
        lin_log_results = sm.OLS(lin_log_endog, lin_log_exog).fit()
        characteristics_dict[key_prefix + '_lin_log_R2' + suffix].append(lin_log_results.rsquared)
        characteristics_dict[key_prefix + '_lin_log_intercept' + suffix].append(lin_log_results.params[0])
        characteristics_dict[key_prefix + '_lin_log_slope' + suffix].append(lin_log_results.params[1])
        characteristics_dict[key_prefix + '_lin_log_intercept_std' + suffix].append(lin_log_results.bse[0])
        characteristics_dict[key_prefix + '_lin_log_slope_std' + suffix].append(lin_log_results.bse[1])
        characteristics_dict[key_prefix + '_lin_log_intercept_p_value' + suffix].append(lin_log_results.pvalues[0])
        characteristics_dict[key_prefix + '_lin_log_slope_p_value' + suffix].append(lin_log_results.pvalues[1])
        R2s.append(lin_log_results.rsquared)
    else:
        characteristics_dict[key_prefix + '_lin_log_R2' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_log_intercept' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_log_slope' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_log_intercept_std' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_log_slope_std' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_log_intercept_p_value' + suffix].append('NA')
        characteristics_dict[key_prefix + '_lin_log_slope_p_value' + suffix].append('NA')
        R2s.append(-1)

    if min(endog) > 0 and min(exog) > 0:
        log_log_exog = sm.add_constant(np.log(exog))
        log_log_endog = np.log(endog)
        log_log_results = sm.OLS(log_log_endog, log_log_exog).fit()
        characteristics_dict[key_prefix + '_log_log_R2' + suffix].append(log_log_results.rsquared)
        characteristics_dict[key_prefix + '_log_log_intercept' + suffix].append(log_log_results.params[0])
        characteristics_dict[key_prefix + '_log_log_slope' + suffix].append(log_log_results.params[1])
        characteristics_dict[key_prefix + '_log_log_intercept_std' + suffix].append(log_log_results.bse[0])
        characteristics_dict[key_prefix + '_log_log_slope_std' + suffix].append(log_log_results.bse[1])
        characteristics_dict[key_prefix + '_log_log_intercept_p_value' + suffix].append(log_log_results.pvalues[0])
        characteristics_dict[key_prefix + '_log_log_slope_p_value' + suffix].append(log_log_results.pvalues[1])
        R2s.append(log_log_results.rsquared)
    else:
        characteristics_dict[key_prefix + '_log_log_R2' + suffix].append('NA')
        characteristics_dict[key_prefix + '_log_log_intercept' + suffix].append('NA')
        characteristics_dict[key_prefix + '_log_log_slope' + suffix].append('NA')
        characteristics_dict[key_prefix + '_log_log_intercept_std' + suffix].append('NA')
        characteristics_dict[key_prefix + '_log_log_slope_std' + suffix].append('NA')
        characteristics_dict[key_prefix + '_log_log_intercept_p_value' + suffix].append('NA')
        characteristics_dict[key_prefix + '_log_log_slope_p_value' + suffix].append('NA')
        R2s.append(-1)

    best_R2_id = np.argmax(R2s)
    best_R2 = R2s[best_R2_id]

    characteristics_dict[key_prefix + '_best_type' + suffix].append(best_R2_id)
    characteristics_dict[key_prefix + '_best_R2' + suffix].append(best_R2)


def init_variance_metrics_dict(characteristics_dict, key_prefix, suffix):
    characteristics_dict[key_prefix + '_lin_lin_R2' + suffix] = []
    characteristics_dict[key_prefix + '_lin_lin_intercept' + suffix] = []
    characteristics_dict[key_prefix + '_lin_lin_slope' + suffix] = []
    characteristics_dict[key_prefix + '_lin_lin_intercept_std' + suffix] = []
    characteristics_dict[key_prefix + '_lin_lin_slope_std' + suffix] = []
    characteristics_dict[key_prefix + '_lin_lin_intercept_p_value' + suffix] = []
    characteristics_dict[key_prefix + '_lin_lin_slope_p_value' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_R2' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_intercept' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_slope' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_intercept_std' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_slope_std' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_intercept_p_value' + suffix] = []
    characteristics_dict[key_prefix + '_lin_log_slope_p_value' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_R2' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_intercept' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_slope' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_intercept_std' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_slope_std' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_intercept_p_value' + suffix] = []
    characteristics_dict[key_prefix + '_log_log_slope_p_value' + suffix] = []
    characteristics_dict[key_prefix + '_best_type' + suffix] = []
    characteristics_dict[key_prefix + '_best_R2' + suffix] = []

    if 'best_type' not in characteristics_dict:
        characteristics_dict['best_type' + suffix] = []
    if 'best_R2' not in characteristics_dict:
        characteristics_dict['best_R2' + suffix] = []


def process_variance(x, y, semi_window, box_b, box_t, metrics_dict, suffix):
    xs, bs, ms, ts = process_box(x, y, semi_window, box_b, box_t)
    process_line_fit(xs, bs, metrics_dict, 'box_b', suffix)
    process_line_fit(xs, ms, metrics_dict, 'box_m', suffix)
    process_line_fit(xs, ts, metrics_dict, 'box_t', suffix)
    R2 = np.min([metrics_dict['box_b_best_R2' + suffix][-1], metrics_dict['box_t_best_R2' + suffix][-1]])
    metrics_dict['best_R2' + suffix].append(R2)
    return xs, bs, ms, ts


def fit_variance(xs, metrics_dict, suffix):
    ys_t = np.zeros(len(xs), dtype=float)
    ys_b = np.zeros(len(xs), dtype=float)

    if metrics_dict['box_b_best_type' + suffix][-1] == 0:  # lin-lin axes
        intercept_box_b = metrics_dict['box_b_lin_lin_intercept' + suffix][-1]
        slope_box_b = metrics_dict['box_b_lin_lin_slope' + suffix][-1]
        if intercept_box_b != 'NA' and slope_box_b != 'NA':
            for x_id in range(0, len(xs)):
                ys_b[x_id] = slope_box_b * xs[x_id] + intercept_box_b

    elif metrics_dict['box_b_best_type' + suffix][-1] == 1:  # lin-log axes
        intercept_box_b = metrics_dict['box_b_lin_log_intercept' + suffix][-1]
        slope_box_b = metrics_dict['box_b_lin_log_slope' + suffix][-1]
        for x_id in range(0, len(xs)):
            ys_b[x_id] = np.exp(slope_box_b * xs[x_id] + intercept_box_b)

    elif metrics_dict['box_b_best_type' + suffix][-1] == 2:  # log-log axes
        intercept_box_b = metrics_dict['box_b_log_log_intercept' + suffix][-1]
        slope_box_b = metrics_dict['box_b_log_log_slope' + suffix][-1]
        for x_id in range(0, len(xs)):
            ys_b[x_id] = np.exp(slope_box_b * np.log(xs[x_id]) + intercept_box_b)

    if metrics_dict['box_t_best_type' + suffix][-1] == 0:  # lin-lin axes
        intercept_box_t = metrics_dict['box_t_lin_lin_intercept' + suffix][-1]
        slope_box_t = metrics_dict['box_t_lin_lin_slope' + suffix][-1]
        for x_id in range(0, len(xs)):
            ys_t[x_id] = slope_box_t * xs[x_id] + intercept_box_t

    elif metrics_dict['box_t_best_type' + suffix][-1] == 1:  # lin-log axes
        intercept_box_t = metrics_dict['box_t_lin_log_intercept' + suffix][-1]
        slope_box_t = metrics_dict['box_t_lin_log_slope' + suffix][-1]
        for x_id in range(0, len(xs)):
            ys_t[x_id] = np.exp(slope_box_t * xs[x_id] + intercept_box_t)

    elif metrics_dict['box_t_best_type' + suffix][-1] == 2:  # log-log axes
        intercept_box_t = metrics_dict['box_t_log_log_intercept' + suffix][-1]
        slope_box_t = metrics_dict['box_t_log_log_slope' + suffix][-1]
        for x_id in range(0, len(xs)):
            ys_t[x_id] = np.exp(slope_box_t * np.log(xs[x_id]) + intercept_box_t)

    return ys_b, ys_t
