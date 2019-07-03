import abc
from pydnameth.config.experiment.types import DataType, Method
import plotly.graph_objs as go
from statsmodels.stats.multitest import multipletests
import plotly.figure_factory as ff
import pydnameth.routines.plot.functions as plot_routines


class ReleaseStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def release(self, config, configs_child):
        pass


class TableReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals_common,
                                      DataType.residuals_special,
                                      DataType.epimutations,
                                      DataType.entropy,
                                      DataType.cells]:

            if config.experiment.method in [Method.z_test_linreg, Method.ancova]:
                reject, pvals_corr, alphacSidak, alphacBonf = multipletests(config.metrics['p_value'],
                                                                            0.05,
                                                                            method='fdr_bh')
                config.metrics['p_value_fdr'] = pvals_corr


class ClockReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):
        pass


class PlotReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):

        if config.experiment.data in [
            DataType.betas,
            DataType.betas_adj,
            DataType.residuals_common,
            DataType.residuals_special,
        ]:
            if config.experiment.method == Method.scatter:

                for item_id, item in enumerate(config.experiment_data['item']):

                    if item in config.cpg_gene_dict:
                        aux = config.cpg_gene_dict[item]
                        if isinstance(aux, list):
                            aux_str = ';'.join(aux)
                        else:
                            aux_str = str(aux)
                    else:
                        aux_str = 'non-genic'

                    layout = plot_routines.get_layout(config, item + '(' + aux_str + ')')

                    raw_item_id = config.experiment.method_params['items'].index(item)

                    if 'x_ranges' in config.experiment.method_params:
                        x_range = config.experiment.method_params['x_ranges'][raw_item_id]
                        if x_range != 'auto' or 'auto' not in x_range:
                            layout.xaxis.range = x_range

                    if 'y_ranges' in config.experiment.method_params:
                        y_range = config.experiment.method_params['y_ranges'][raw_item_id]
                        if y_range != 'auto' or 'auto' not in y_range:
                            layout.yaxis.range = y_range

                    fig = go.Figure(data=config.experiment_data['data'][item_id], layout=layout)
                    config.experiment_data['fig'].append(fig)

            elif config.experiment.method == Method.variance_histogram:

                for data in config.experiment_data['data']:

                    layout = plot_routines.get_layout(config)
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

                    config.experiment_data['fig'].append(fig)

            elif config.experiment.method == Method.curve:

                layout = plot_routines.get_layout(config)

                config.experiment_data['fig'] = go.Figure(data=config.experiment_data['data'], layout=layout)

        elif config.experiment.data in [
            DataType.entropy,
            DataType.cells,
        ]:
            if config.experiment.method == Method.scatter:

                layout = plot_routines.get_layout(config)

                if 'x_range' in config.experiment.method_params:
                    x_range = config.experiment.method_params['x_range']
                    if x_range != 'auto' or 'auto' not in x_range:
                        layout.xaxis.range = x_range

                if 'y_range' in config.experiment.method_params:
                    y_range = config.experiment.method_params['y_range']
                    if y_range != 'auto' or 'auto' not in y_range:
                        layout.yaxis.range = y_range

                fig = go.Figure(data=config.experiment_data['data'], layout=layout)
                config.experiment_data['fig'] = fig

        elif config.experiment.data == DataType.epimutations:

            if config.experiment.method == Method.scatter:

                layout = plot_routines.get_layout(config)

                if config.experiment.method_params['x_range'] != 'auto':
                    layout.xaxis.range = config.experiment.method_params['x_range']

                if config.experiment.method_params['y_range'] != 'auto':
                    layout.yaxis.range = config.experiment.method_params['y_range']

                layout.yaxis.type = config.experiment.method_params['y_type']
                if layout.yaxis.type == 'log':
                    layout.yaxis.tickvals = [1, 2, 5,
                                             10, 20, 50,
                                             100, 200, 500,
                                             1000, 2000, 5000,
                                             10000, 20000, 50000,
                                             100000, 200000, 500000]

                config.experiment_data['fig'] = go.Figure(data=config.experiment_data['data'], layout=layout)

            if config.experiment.method == Method.range:

                layout = plot_routines.get_layout(config)

                if config.experiment.method_params['x_range'] != 'auto':
                    layout.xaxis.range = config.experiment.method_params['x_range']

                borders = config.experiment.method_params['borders']

                labels = []
                tickvals = []
                for seg_id in range(0, len(borders) - 1):
                    x_center = (borders[seg_id + 1] + borders[seg_id]) * 0.5
                    tickvals.append(x_center)
                    labels.append(f'{borders[seg_id]} to {borders[seg_id + 1] - 1}')
                layout.xaxis.tickvals = tickvals
                layout.xaxis.ticktext = labels

                if config.experiment.method_params['y_range'] != 'auto':
                    layout.yaxis.range = config.experiment.method_params['y_range']

                layout.yaxis.type = config.experiment.method_params['y_type']
                if layout.yaxis.type == 'log':
                    layout.yaxis.tickvals = [1, 2, 5,
                                             10, 20, 50,
                                             100, 200, 500,
                                             1000, 2000, 5000,
                                             10000, 20000, 50000,
                                             100000, 200000, 500000]

                config.experiment_data['fig'] = go.Figure(data=config.experiment_data['data'], layout=layout)

        elif config.experiment.data == DataType.observables:

            if config.experiment.method == Method.histogram:

                layout = plot_routines.get_layout(config)

                if 'x_range' in config.experiment.method_params:
                    if config.experiment.method_params['x_range'] != 'auto':
                        layout.xaxis.range = config.experiment.method_params['x_range']

                config.experiment_data['fig'] = go.Figure(data=config.experiment_data['data'], layout=layout)


class CreateReleaseStrategy(ReleaseStrategy):

    def release(self, config, configs_child):
        pass
