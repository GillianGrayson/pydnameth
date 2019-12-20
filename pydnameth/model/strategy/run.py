import abc
from pydnameth.config.experiment.types import Method, DataType
from pydnameth.config.experiment.metrics import get_method_metrics_keys
import numpy as np
from pydnameth.routines.clock.types import ClockExogType, Clock
from pydnameth.routines.clock.linreg.processing import build_clock_linreg
import plotly.graph_objs as go
import colorlover as cl
from pydnameth.routines.common import is_float, get_names
from tqdm import tqdm
from pydnameth.routines.variance.functions import process_variance, fit_variance, get_box_xs
from pydnameth.routines.common import update_parent_dict_with_children
from pydnameth.routines.heteroscedasticity.functions import process_heteroscedasticity
from pydnameth.routines.linreg.functions import process_linreg
from pydnameth.routines.z_test_slope.functions import process_z_test_slope
from pydnameth.routines.polygon.functions import process_linreg_polygon, process_variance_polygon
from pydnameth.routines.cluster.functions import process_cluster
from pydnameth.routines.plot.functions.scatter import process_scatter
from pydnameth.routines.plot.functions.range import process_range
from pydnameth.routines.plot.functions.variance_histogram import process_variance_histogram
import string
import pandas as pd
from statsmodels.formula.api import ols
from scipy.stats import pearsonr


class RunStrategy(metaclass=abc.ABCMeta):

    def __init__(self, get_strategy):
        self.get_strategy = get_strategy

    @abc.abstractmethod
    def single(self, item, config, configs_child):
        pass

    @abc.abstractmethod
    def iterate(self, config, configs_child):
        pass

    @abc.abstractmethod
    def run(self, config, configs_child):
        pass


class TableRunStrategy(RunStrategy):

    def single(self, item, config, configs_child):

        if config.experiment.method == Method.heteroskedasticity:

            x = self.get_strategy.get_target(config, item)
            y = self.get_strategy.get_single_base(config, item)
            process_heteroscedasticity(x, y, config.metrics, f'_{config.hash[0:8]}')

        elif config.experiment.method == Method.linreg:

            x = self.get_strategy.get_target(config, item)
            y = self.get_strategy.get_single_base(config, item)
            process_linreg(x, y, config.metrics, f'_{config.hash[0:8]}')

        elif config.experiment.method == Method.cluster:

            x = self.get_strategy.get_target(config, item)
            y = self.get_strategy.get_single_base(config, item)
            process_cluster(x, y, config.experiment.method_params, config.metrics, f'_{config.hash[0:8]}')

        elif config.experiment.method == Method.oma:

            x = self.get_strategy.get_target(config, item)
            y = self.get_strategy.get_single_base(config, item)

            corr_coeff, p_value = pearsonr(x, y)

            config.metrics['corr_coeff' + f'_{config.hash[0:8]}'].append(corr_coeff)
            config.metrics['p_value' + f'_{config.hash[0:8]}'].append(p_value)

        elif config.experiment.method == Method.polygon:

            xs = []
            ys = []
            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)
                x = self.get_strategy.get_target(config_child, item)
                y = self.get_strategy.get_single_base(config_child, item)
                xs.append(x)
                ys.append(y)

            if config.experiment.method_params['method'] == Method.linreg:
                process_linreg_polygon(configs_child, item, xs, config.metrics, f'_{config.hash[0:8]}')

            elif config.experiment.method_params['method'] == Method.variance:
                process_variance_polygon(configs_child, item, xs, config.metrics, f'_{config.hash[0:8]}')

        elif config.experiment.method == Method.z_test_linreg:

            slopes = []
            slopes_std = []
            num_subs = []
            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)
                item_id = config_child.advanced_dict[item]
                slopes.append(config_child.advanced_data['slope' + f'_{config_child.hash[0:8]}'][item_id])
                slopes_std.append(config_child.advanced_data['slope_std' + f'_{config_child.hash[0:8]}'][item_id])
                num_subs.append(len(config_child.observables_dict['age']))

            process_z_test_slope(slopes, slopes_std, num_subs, config.metrics, f'_{config.hash[0:8]}')

        elif config.experiment.method == Method.ancova:

            x_all = []
            y_all = []
            category_all = []
            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)
                x = self.get_strategy.get_target(config_child, item)
                y = self.get_strategy.get_single_base(config_child, item)
                x_all += list(x)
                y_all += list(y)
                category_all += [list(string.ascii_lowercase)[configs_child.index(config_child)]] * len(x)

            data = {'x': x_all, 'y': y_all, 'category': category_all}
            df = pd.DataFrame(data)
            formula = 'y ~ x * category'
            lm = ols(formula, df)
            results = lm.fit()
            p_value = results.pvalues[3]

            config.metrics['p_value' + f'_{config.hash[0:8]}'].append(p_value)

        elif config.experiment.method == Method.aggregator:

            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)

        elif config.experiment.method == Method.variance:

            x = self.get_strategy.get_target(config, item)
            y = self.get_strategy.get_single_base(config, item)

            semi_window = config.experiment.method_params['semi_window']
            box_b = config.experiment.method_params['box_b']
            box_t = config.experiment.method_params['box_t']

            process_variance(x, y, semi_window, box_b, box_t, config.metrics, f'_{config.hash[0:8]}')

            xs = get_box_xs(x)
            ys_b, ys_t = fit_variance(xs, config.metrics, f'_{config.hash[0:8]}')

            diff_begin = abs(ys_t[0] - ys_b[0])
            diff_end = abs(ys_t[-1] - ys_b[-1])

            config.metrics['increasing_div' + f'_{config.hash[0:8]}'].append(
                max(diff_begin, diff_end) / min(diff_begin, diff_end)
            )
            config.metrics['increasing_sub' + f'_{config.hash[0:8]}'].append(
                abs(diff_begin - diff_end)
            )

        config.metrics['item'].append(item)
        aux = self.get_strategy.get_aux(config, item)
        config.metrics['aux'].append(aux)

    def iterate(self, config, configs_child):

        for item in tqdm(config.base_list, mininterval=60.0, desc=f'{str(config.experiment)} running'):
            if item in config.base_dict:
                self.single(item, config, configs_child)

    def run(self, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals,
                                      DataType.epimutations,
                                      DataType.entropy,
                                      DataType.cells,
                                      DataType.genes]:

            self.iterate(config, configs_child)


class ClockRunStrategy(RunStrategy):

    def single(self, item, config, configs_child):
        pass

    def iterate(self, config, configs_child):
        pass

    def run(self, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals]:

            if config.experiment.method == Method.linreg:

                items = config.experiment_data['items']
                values = config.experiment_data['values']
                test_size = config.experiment_data['test_size']
                train_size = config.experiment_data['train_size']

                target = self.get_strategy.get_target(config)

                type = config.experiment.method_params['type']
                runs = config.experiment.method_params['runs']
                size = min(config.experiment.method_params['size'], train_size, len(items))
                config.experiment.method_params['size'] = size

                if type == ClockExogType.all.value:

                    for exog_id in tqdm(range(0, size), mininterval=60.0, desc=f'clock building'):
                        config.metrics['item'].append(items[exog_id])
                        aux = self.get_strategy.get_aux(config, items[exog_id])
                        config.metrics['aux'].append(aux)

                        clock = Clock(
                            endog_data=target,
                            endog_names=config.attributes.target,
                            exog_data=values[0:exog_id + 1],
                            exog_names=items[0:exog_id + 1],
                            metrics_dict=config.metrics,
                            train_size=train_size,
                            test_size=test_size,
                            exog_num=exog_id + 1,
                            exog_num_comb=exog_id + 1,
                            num_bootstrap_runs=runs
                        )

                        build_clock_linreg(clock, f'_{config.hash[0:8]}')

                elif type == ClockExogType.deep.value:

                    for exog_id in tqdm(range(0, size), mininterval=60.0, desc=f'clock building'):
                        config.metrics['item'].append(exog_id + 1)
                        config.metrics['aux'].append(exog_id + 1)

                        clock = Clock(
                            endog_data=target,
                            endog_names=config.attributes.target,
                            exog_data=values[0:size + 1],
                            exog_names=items[0:size + 1],
                            metrics_dict=config.metrics,
                            train_size=train_size,
                            test_size=test_size,
                            exog_num=size,
                            exog_num_comb=exog_id + 1,
                            num_bootstrap_runs=runs
                        )

                        build_clock_linreg(clock, f'_{config.hash[0:8]}')

                elif type == ClockExogType.single.value:

                    config.metrics['item'].append(size)
                    config.metrics['aux'].append(size)

                    clock = Clock(
                        endog_data=target,
                        endog_names=config.attributes.target,
                        exog_data=values[0:size],
                        exog_names=items[0:size],
                        metrics_dict=config.metrics,
                        train_size=train_size,
                        test_size=test_size,
                        exog_num=size,
                        exog_num_comb=size,
                        num_bootstrap_runs=runs
                    )

                    build_clock_linreg(clock, f'_{config.hash[0:8]}')


class PlotRunStrategy(RunStrategy):

    def single(self, item, config, configs_child):

        xs = []
        ys = []
        names = []
        for config_child in configs_child:
            xs.append(self.get_strategy.get_target(config_child, item))
            ys.append(self.get_strategy.get_single_base(config_child, item))
            names.append(get_names(config_child, config.experiment.method_params))

        if config.experiment.method == Method.scatter:
            process_scatter(config.experiment_data['data'], config.experiment.method_params, xs, ys, names)

        elif config.experiment.method == Method.range:
            process_range(config.experiment_data['data'], config.experiment.method_params, xs, ys)

        elif config.experiment.method == Method.variance_histogram:
            process_variance_histogram(config.experiment_data['data'], xs, ys, names)

    def iterate(self, config, configs_child):
        items = config.experiment.method_params['items']
        for item in items:
            if item in config.base_dict:
                print(item)
                config.experiment_data['item'].append(item)
                self.single(item, config, configs_child)
            else:
                if 'type' in config.experiment.task_params:
                    if config.experiment.task_params['type'] == 'prepare':
                        config.experiment_data['data'].append([])

    def run(self, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals,
                                      DataType.epimutations,
                                      DataType.entropy,
                                      DataType.cells,
                                      DataType.genes]:

            if config.experiment.method in [Method.scatter, Method.variance_histogram]:
                self.iterate(config, configs_child)

            elif config.experiment.method == Method.curve:

                x_target = config.experiment.method_params['x']
                y_target = config.experiment.method_params['y']
                number_of_points = int(config.experiment.method_params['number_of_points'])

                plot_data = []

                for config_child in configs_child:

                    if x_target == 'count':
                        xs = list(range(1, number_of_points + 1))
                    else:
                        if x_target in config_child.advanced_data:
                            xs = config_child.advanced_data[x_target][0:number_of_points]
                        else:
                            raise ValueError(f'{x_target} not in {config_child}.')

                    if y_target in config_child.advanced_data:
                        ys = config_child.advanced_data[y_target][0:number_of_points]
                    else:
                        raise ValueError(f'{y_target} not in {config_child}.')

                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]
                    coordinates = color[4:-1].split(',')
                    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'
                    color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.9) + ')'

                    scatter = go.Scatter(
                        x=xs,
                        y=ys,
                        name=get_names(config_child, config.experiment.method_params),
                        mode='lines+markers',
                        marker=dict(
                            size=10,
                            color=color_transparent,
                            line=dict(
                                width=2,
                                color=color_border,
                            )
                        ),
                    )
                    plot_data.append(scatter)

                config.experiment_data['data'] = plot_data

        elif config.experiment.data == DataType.observables:

            if config.experiment.method == Method.histogram:

                plot_data = []
                num_points = []
                for config_child in configs_child:

                    curr_plot_data = []

                    targets = self.get_strategy.get_target(config_child)
                    num_points.append(len(targets))
                    is_number_list = [is_float(t) for t in targets]
                    if False in is_number_list:
                        xbins = {}
                    else:
                        bin_size = config.experiment.method_params['bin_size']
                        xbins = dict(
                            start=min(targets) - 0.5 * bin_size,
                            end=max(targets) + 0.5 * bin_size,
                            size=bin_size
                        )

                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]

                    if config_child.experiment.method == Method.histogram:
                        histogram = go.Histogram(
                            x=targets,
                            name=get_names(config_child, config.experiment.method_params),
                            xbins=xbins,
                            marker=dict(
                                opacity=config.experiment.method_params['opacity'],
                                color=color,
                                line=dict(
                                    color='#444444',
                                    width=1
                                )
                            )
                        )

                        curr_plot_data.append(histogram)

                    plot_data += curr_plot_data

                # Sorting by total number of points
                order = np.argsort(num_points)[::-1]
                config.experiment_data['data'] = [plot_data[index] for index in order]


class CreateRunStrategy(RunStrategy):

    def single(self, item, config_child, configs_child):
        pass

    def iterate(self, config, configs_child):
        pass

    def run(self, config, configs_child):
        pass
