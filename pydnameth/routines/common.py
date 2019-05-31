import plotly.graph_objs as go
import numpy as np


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def find_nearest_id(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def dict_slice(origin_dict, id):
    new_dict = {}
    for key, value in origin_dict.items():
        new_dict[key] = [value[id]]
    return new_dict


def normalize_to_0_1(values):
    values_normed = [(float(val) - min(values)) / (float(max(values)) - float(min(values))) for val in values]
    return values_normed


def get_axis(title):
    axis = dict(
        title=title,
        showgrid=True,
        showline=True,
        mirror='ticks',
        titlefont=dict(
            family='Arial',
            size=33,
            color='black'
        ),
        showticklabels=True,
        tickangle=0,
        tickfont=dict(
            family='Arial',
            size=30,
            color='black'
        ),
        exponentformat='e',
        showexponent='all'
    )
    return axis


def get_legend():
    legend = dict(
        font=dict(
            family='Arial',
            size=33,
        ),
        orientation="h",
        x=0.25,
        y=1.2,
    )
    return legend


def get_margin():
    margin = go.layout.Margin(
        l=80,
        r=20,
        b=80,
        t=10,
        pad=0
    )
    return margin


def process_names(config):
    name = str(config.attributes.observables)
    if 'gender' in name:
        name = name.replace('gender', 'sex')
    return name


def get_names(config, plot_params):
    if 'legend_size' in plot_params:
        legend_size = plot_params['legend_size']
        parts = process_names(config).split(')_')
        if len(parts) > 1:
            name = ')_'.join(parts[0:legend_size]) + ')'
        elif legend_size > len(parts):
            name = process_names(config)
        else:
            name = process_names(config)
    else:
        name = process_names(config)
    return name
