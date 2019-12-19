import abc
from pydnameth.config.experiment.types import DataType, Method
import plotly.graph_objs as go
from statsmodels.stats.multitest import multipletests
import plotly.figure_factory as ff
from pydnameth.routines.plot.functions.layout import get_layout
from pydnameth.routines.common import get_axis


class ReleaseStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def release(self, config, configs_child):
        pass


class TableReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals,
                                      DataType.epimutations,
                                      DataType.entropy,
                                      DataType.cells]:

            if config.experiment.method in [Method.ancova, Method.oma]:
                reject, pvals_corr, alphacSidak, alphacBonf = multipletests(config.metrics['p_value'],
                                                                            0.05,
                                                                            method='fdr_bh')
                config.metrics['p_value_fdr_bh' + f'_{config.hash[0:8]}'] = pvals_corr

                reject, pvals_corr, alphacSidak, alphacBonf = multipletests(config.metrics['p_value'],
                                                                            0.05,
                                                                            method='bonferroni')
                config.metrics['p_value_bonferroni' + f'_{config.hash[0:8]}'] = pvals_corr


class ClockReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):
        pass


class PlotReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):

        if config.experiment.task_params is None or config.experiment.task_params['type'] == 'run':

            if config.experiment.data in [
                DataType.betas,
                DataType.betas_adj,
                DataType.residuals,
                DataType.epimutations,
                DataType.entropy,
                DataType.cells,
                DataType.genes
            ]:
                if config.experiment.method in [Method.scatter, Method.range]:

                    for item_id, items in enumerate(config.experiment_data['item']):

                        if config.experiment.data in [
                            DataType.betas,
                            DataType.betas_adj,
                            DataType.residuals,
                        ]:
                            if items in config.cpg_gene_dict:
                                aux = config.cpg_gene_dict[items]
                                if isinstance(aux, list):
                                    aux_str = ';'.join(aux)
                                else:
                                    aux_str = str(aux)
                            else:
                                aux_str = 'non-genic'
                            title = items + '(' + aux_str + ')'
                        elif config.experiment.data == DataType.genes:
                            title = items
                        else:
                            title = ''

                        if config.experiment.method == Method.range:
                            layout = get_layout(config)
                        else:
                            layout = get_layout(config, title)

                        if config.experiment.data == DataType.cells:
                            layout.yaxis = get_axis(items)

                        raw_item_id = config.experiment.method_params['items'].index(items)

                        if 'x_ranges' in config.experiment.method_params:
                            x_range = config.experiment.method_params['x_ranges'][raw_item_id]
                            if x_range != 'auto' or 'auto' not in x_range:
                                layout.xaxis.range = x_range

                        if 'y_ranges' in config.experiment.method_params:
                            y_range = config.experiment.method_params['y_ranges'][raw_item_id]
                            if y_range != 'auto' or 'auto' not in y_range:
                                layout.yaxis.range = y_range

                        if config.experiment.method == Method.range:
                            borders = config.experiment.method_params['borders']
                            labels = []
                            tickvals = []
                            for seg_id in range(0, len(borders) - 1):
                                x_center = (borders[seg_id + 1] + borders[seg_id]) * 0.5
                                tickvals.append(x_center)
                                labels.append(f'{borders[seg_id]} to {borders[seg_id + 1] - 1}')
                            layout.xaxis.tickvals = tickvals
                            layout.xaxis.ticktext = labels

                        fig = go.Figure(data=config.experiment_data['data'][item_id], layout=layout)
                        config.experiment_data['fig'].append(fig)

                elif config.experiment.method == Method.scatter_comparison:

                    x_num = len(configs_child)
                    if x_num == 3:
                        x_begin = 0.11
                    elif x_num == 2:
                        x_begin = 0.2
                    else:
                        x_begin = 0.075
                    x_end = 1
                    x_shift = (x_end - x_begin) / x_num
                    x_size = x_shift - 0.01
                    x_domains = []
                    for x_id in range(0, x_num):
                        x = x_begin + x_shift * x_id
                        x_domains.append([x, x + x_size])

                    y_num = len(configs_child[0].experiment_data['item'])

                    if y_num == 1:
                        y_begin = 0.25
                    elif y_num == 2:
                        y_begin = 0.2
                    else:
                        y_begin = 0.06

                    y_end = 1
                    y_shift = (y_end - y_begin) / y_num
                    y_size = y_shift - 0.02
                    y_domains = []
                    for y_id in range(0, y_num):
                        y = y_begin + y_shift * y_id
                        y_domains.append([y, y + y_size])

                    for configs_child_id, config_child in enumerate(configs_child):
                        for item_id, items in enumerate(config_child.experiment_data['data']):

                            if configs_child_id == 0:
                                x_string = 'x'
                            else:
                                x_string = f'x{configs_child_id + 1}'

                            if item_id == 0:
                                y_string = 'y'
                            else:
                                y_string = f'y{item_id + 1}'

                            if isinstance(items, list):
                                for item in items:
                                    item.xaxis = x_string
                                    item.yaxis = y_string

                                    if item.mode == 'markers':
                                        item.marker.size = 1.5
                                        item.marker.line.width = 0.2
                                    if item.mode == 'lines':
                                        item.line.width = 1

                                    config.experiment_data['data'].append(item)
                            else:
                                items.xaxis = x_string
                                items.yaxis = y_string

                                if items.mode == 'markers':
                                    items.marker.size = 1
                                    items.marker.line.width = 0.2
                                if items.mode == 'lines':
                                    items.line.width = 2

                                config.experiment_data['data'].append(items)

                    layout = {}
                    layout['showlegend'] = False
                    layout['margin'] = {
                        'l': 0,
                        'r': 0,
                        'b': 0,
                        't': 0,
                    }
                    height_per_row = 125
                    width_per_col = 200
                    layout['height'] = height_per_row * y_num
                    layout['width'] = width_per_col * x_num

                    for x_id in range(0, x_num):

                        if x_id == 0:
                            x_string_add = ''
                        else:
                            x_string_add = str(x_id + 1)

                        layout['xaxis' + x_string_add] = {}
                        layout['xaxis' + x_string_add]['domain'] = x_domains[x_id]
                        layout['xaxis' + x_string_add]['anchor'] = 'x' + x_string_add

                        layout['xaxis' + x_string_add]['zeroline'] = False
                        layout['xaxis' + x_string_add]['showgrid'] = True
                        layout['xaxis' + x_string_add]['showline'] = True
                        layout['xaxis' + x_string_add]['mirror'] = 'allticks'

                        layout['xaxis' + x_string_add]['titlefont'] = dict(
                            family='Arial',
                            size=13,
                            color='black'
                        )

                        layout['xaxis' + x_string_add]['tickfont'] = dict(
                            family='Arial',
                            size=10,
                            color='black'
                        )

                        db = config.experiment.method_params['data_bases'][x_id]
                        x_title = db
                        layout['xaxis' + x_string_add]['title'] = x_title

                        x_range = config.experiment.method_params['x_ranges'][x_id]
                        if x_range != 'auto' or 'auto' not in x_range:
                            layout['xaxis' + x_string_add]['range'] = x_range

                    for y_id in range(0, y_num):

                        if y_id == 0:
                            y_string_add = ''
                        else:
                            y_string_add = str(y_id + 1)

                        layout['yaxis' + y_string_add] = {}
                        layout['yaxis' + y_string_add]['domain'] = y_domains[y_id]
                        layout['yaxis' + y_string_add]['anchor'] = 'y' + y_string_add

                        layout['yaxis' + y_string_add]['zeroline'] = False
                        layout['yaxis' + y_string_add]['showgrid'] = True
                        layout['yaxis' + y_string_add]['showline'] = True
                        layout['yaxis' + y_string_add]['mirror'] = 'allticks'

                        layout['yaxis' + y_string_add]['titlefont'] = dict(
                            family='Arial',
                            size=13,
                            color='black'
                        )

                        layout['yaxis' + y_string_add]['tickfont'] = dict(
                            family='Arial',
                            size=10,
                            color='black'
                        )

                        y_title = config.experiment.method_params['items'][y_id]
                        if config.experiment.data in [
                            DataType.betas,
                            DataType.betas_adj,
                            DataType.residuals,
                        ]:
                            if 'aux' in config.experiment.method_params:
                                aux = config.experiment.method_params['aux'][y_id]
                                if aux == '':
                                    aux = 'Non-genic'
                                y_title = y_title + '<br>' + aux
                        layout['yaxis' + y_string_add]['title'] = y_title

                        y_range = config.experiment.method_params['y_ranges'][y_id]
                        if y_range != 'auto' or 'auto' not in y_range:
                            layout['yaxis' + y_string_add]['range'] = y_range

                    fig = go.Figure(data=config.experiment_data['data'], layout=layout)
                    config.experiment_data['fig'] = fig

                elif config.experiment.method == Method.variance_histogram:

                    for data in config.experiment_data['data']:
                        layout = get_layout(config)
                        layout.xaxis.title = '$\\Delta$'
                        layout.yaxis.title = '$PDF$'

                        fig = ff.create_distplot(
                            data['hist_data'],
                            data['group_labels'],
                            show_hist=False,
                            show_rug=False,
                            colors=data['colors']
                        )
                        fig['layout'] = layout

                        config.experiment_data['fig'] = fig

                elif config.experiment.method == Method.curve:

                    layout = get_layout(config)
                    config.experiment_data['fig'] = go.Figure(data=config.experiment_data['data'], layout=layout)

            elif config.experiment.data == DataType.observables:

                if config.experiment.method == Method.histogram:

                    layout = get_layout(config)

                    if 'x_range' in config.experiment.method_params:
                        if config.experiment.method_params['x_range'] != 'auto':
                            layout.xaxis.range = config.experiment.method_params['x_range']

                    config.experiment_data['fig'] = go.Figure(data=config.experiment_data['data'], layout=layout)

        elif config.experiment.task_params['type'] == 'prepare':
            pass


class CreateReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):
        pass
