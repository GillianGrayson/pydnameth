import statsmodels.api as sm
import colorlover as cl


def process_variance_histogram(experiment_data, xs_all, ys_all, names_all):

    plot_data = {
        'hist_data': [],
        'group_labels': [],
        'colors': []
    }

    for child_id in range(0, len(xs_all)):

        plot_data['group_labels'].append(names_all[child_id])
        plot_data['colors'].append(cl.scales['8']['qual']['Set1'][child_id])

        x = sm.add_constant(xs_all[child_id])
        y = ys_all[child_id]

        results = sm.OLS(y, x).fit()

        plot_data['hist_data'].append(results.resid)

    experiment_data.append(plot_data)
