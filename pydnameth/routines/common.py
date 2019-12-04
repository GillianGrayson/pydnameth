import plotly.graph_objs as go
import numpy as np
import pandas


def categorize_data(data):
    can_cast = np.can_cast(data, float)

    if can_cast:
        data = data.astype(float)
    else:
        data = pandas.factorize(data)[0]
        data = np.array(data, dtype=float)

    return data


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
    max_val = max(values)
    min_val = min(values)
    shift_val = (max_val - min_val)
    values_normed = (np.asarray(values) - min_val) / shift_val
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


def update_parent_dict_with_children(parent_metrics_keys, item, config_parent, config_child):
    item_id = config_child.advanced_dict[item]
    for key in config_child.advanced_data:
        if key not in parent_metrics_keys:
            advanced_data = config_child.advanced_data[key][item_id]
            suffix = str(config_child.attributes.observables)
            if suffix != '' and suffix not in key:
                key += '_' + suffix
            prefix = str(config_child.experiment.method) + '_' + config_child.attributes.target
            if prefix != '' and prefix not in key:
                key = prefix + '_' + key
            config_parent.metrics[key].append(advanced_data)
            parent_metrics_keys.append(key)
