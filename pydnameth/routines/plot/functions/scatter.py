import statsmodels.api as sm
import numpy as np
import plotly.graph_objs as go
import colorlover as cl
from pydnameth.routines.polygon.types import PolygonRoutines
from pydnameth.routines.variance.functions import \
    process_box, init_variance_metrics_dict, process_variance, fit_variance


def process_scatter(experiment_data, method_params, xs_all, ys_all, names_all):

    line = method_params['line']
    add = method_params['add']
    fit = method_params['fit']
    semi_window = method_params['semi_window']
    box_b = method_params['box_b']
    box_t = method_params['box_t']

    plot_data = []
    num_points = []
    for child_id in range(0, len(xs_all)):

        curr_plot_data = []

        # Plot data
        targets = xs_all[child_id]
        num_points.append(len(targets))
        data = ys_all[child_id]

        # Colors setup
        color = cl.scales['8']['qual']['Set1'][child_id]
        coordinates = color[4:-1].split(',')
        color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.1) + ')'
        color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

        # Adding scatter
        scatter = go.Scatter(
            x=targets,
            y=data,
            name=names_all[child_id],
            mode='markers',
            marker=dict(
                size=4,
                color=color_border,
                line=dict(
                    width=1,
                    color=color_border,
                )
            ),
        )
        curr_plot_data.append(scatter)

        # Linear regression
        x = sm.add_constant(targets)
        y = data
        results = sm.OLS(y, x).fit()
        intercept = results.params[0]
        slope = results.params[1]
        intercept_std = results.bse[0]
        slope_std = results.bse[1]

        # Adding regression line
        if line == 'yes':
            x_min = np.min(targets)
            x_max = np.max(targets)
            y_min = slope * x_min + intercept
            y_max = slope * x_max + intercept
            scatter = go.Scatter(
                x=[x_min, x_max],
                y=[y_min, y_max],
                mode='lines',
                marker=dict(
                    color=color
                ),
                line=dict(
                    width=6,
                    color=color
                ),
                showlegend=False
            )
            curr_plot_data.append(scatter)

        # Adding polygon area
        if add == 'polygon':
            pr = PolygonRoutines(
                x=targets,
                metrics_dict={
                    'intercept': [intercept],
                    'slope': [slope],
                    'intercept_std': [intercept_std],
                    'slope_std': [slope_std]
                },
                suffix=''
            )
            scatter = pr.get_scatter(color_transparent)
            curr_plot_data.append(scatter)

        # Adding box curve
        if fit == 'no' and semi_window != 'none':

            xs, bs, ms, ts = process_box(targets, data, semi_window, box_b, box_t)

            scatter = go.Scatter(
                x=xs,
                y=bs,
                name=names_all[child_id],
                mode='lines',
                line=dict(
                    width=4,
                    color=color_border
                ),
                showlegend=False
            )
            curr_plot_data.append(scatter)

            scatter = go.Scatter(
                x=xs,
                y=ms,
                name=names_all[child_id],
                mode='lines',
                line=dict(
                    width=6,
                    color=color_border
                ),
                showlegend=False
            )
            curr_plot_data.append(scatter)

            scatter = go.Scatter(
                x=xs,
                y=ts,
                name=names_all[child_id],
                mode='lines',
                line=dict(
                    width=4,
                    color=color_border
                ),
                showlegend=False
            )
            curr_plot_data.append(scatter)

        # Adding best curve
        if fit == 'yes' and semi_window != 'none':

            metrics_dict = {}
            init_variance_metrics_dict(metrics_dict, 'box_b')
            init_variance_metrics_dict(metrics_dict, 'box_m')
            init_variance_metrics_dict(metrics_dict, 'box_t')

            xs, _, _, _ = process_variance(targets, data, semi_window, box_b, box_t, metrics_dict)

            ys_b, ys_t = fit_variance(xs, metrics_dict)

            scatter = go.Scatter(
                x=xs,
                y=ys_t,
                name=names_all[child_id],
                mode='lines',
                line=dict(
                    width=4,
                    color=color_border
                ),
                showlegend=False
            )
            curr_plot_data.append(scatter)

            scatter = go.Scatter(
                x=xs,
                y=ys_b,
                name=names_all[child_id],
                mode='lines',
                line=dict(
                    width=4,
                    color=color_border
                ),
                showlegend=False
            )
            curr_plot_data.append(scatter)

        plot_data.append(curr_plot_data)

    # Sorting by total number of points
    order = np.argsort(num_points)[::-1]
    curr_data = []
    for index in order:
        curr_data += plot_data[index]
    experiment_data.append(curr_data)
