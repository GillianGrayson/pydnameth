import plotly.graph_objs as go
import colorlover as cl


def process_range(experiment_data, method_params, xs_all, ys_all):

    borders = method_params['borders']

    plot_data = []

    for child_id in range(0, len(xs_all)):

        color = cl.scales['8']['qual']['Set1'][child_id]
        coordinates = color[4:-1].split(',')
        color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'

        x = xs_all[child_id]
        y = ys_all[child_id]

        for seg_id in range(0, len(borders) - 1):
            x_center = (borders[seg_id + 1] + borders[seg_id]) * 0.5
            curr_box = []
            for subj_id in range(0, len(xs_all)):
                if borders[seg_id] <= x[subj_id] < borders[seg_id + 1]:
                    curr_box.append(y[subj_id])

            trace = go.Box(
                y=curr_box,
                x=[x_center] * len(curr_box),
                name=f'{borders[seg_id]} to {borders[seg_id + 1] - 1}',
                marker=dict(
                    color=color_transparent
                )
            )
            plot_data.append(trace)

    experiment_data.append(plot_data)
