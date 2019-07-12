import abc
from pydnameth.config.experiment.types import Method, DataType
from pydnameth.config.experiment.metrics import get_method_metrics_keys
import statsmodels.api as sm
import numpy as np
from sklearn.cluster import DBSCAN
from pydnameth.routines.clock.types import ClockExogType, Clock
from pydnameth.routines.clock.linreg.processing import build_clock_linreg
import plotly.graph_objs as go
import colorlover as cl
from shapely import geometry
from pydnameth.routines.common import is_float, get_names, normalize_to_0_1
from pydnameth.routines.polygon.types import PolygonRoutines
from tqdm import tqdm
from pydnameth.routines.variance.functions import \
    process_box, init_variance_metrics_dict, process_variance, fit_variance, get_box_xs
from pydnameth.routines.common import find_nearest_id, dict_slice, update_parent_dict_with_children
from pydnameth.routines.linreg.functions import process_linreg
from pydnameth.routines.z_test_slope.functions import z_test_slope_proc
from pydnameth.routines.polygon.functions import process_linreg_polygon, process_variance_polygon
import string
import pandas as pd
from statsmodels.formula.api import ols


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

        if config.experiment.method == Method.linreg:

            targets = self.get_strategy.get_target(config)
            x = sm.add_constant(targets)
            y = self.get_strategy.get_single_base(config, [item])[0]

            process_linreg(x, y, config.metrics)

        elif config.experiment.method == Method.cluster:

            x = self.get_strategy.get_target(config)
            x_normed = normalize_to_0_1(x)
            y = self.get_strategy.get_single_base(config, [item])[0]
            y_normed = normalize_to_0_1(y)

            min_samples = max(1, int(config.experiment.method_params['min_samples_percentage'] * len(x) / 100.0))

            X = np.array([x_normed, y_normed]).T
            db = DBSCAN(eps=config.experiment.method_params['eps'], min_samples=min_samples).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            number_of_noise_points = list(labels).count(-1)
            percent_of_noise_points = float(number_of_noise_points) / float(len(x)) * 100.0

            config.metrics['number_of_clusters'].append(number_of_clusters)
            config.metrics['number_of_noise_points'].append(number_of_noise_points)
            config.metrics['percent_of_noise_points'].append(percent_of_noise_points)

        elif config.experiment.method == Method.polygon:

            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)

            xs = []
            ys = []
            for config_child in configs_child:
                x = np.squeeze(np.asarray(self.get_strategy.get_target(config_child)))
                y = np.squeeze(np.asarray(self.get_strategy.get_single_base(config_child, [item])))
                xs.append(x)
                ys.append(y)

            if config.experiment.method_params['method'] == Method.linreg:
                process_linreg_polygon(configs_child, item, xs, config.metrics)

            elif config.experiment.method_params['method'] == Method.variance:
                process_variance_polygon(configs_child, item, xs, config.metrics)

        elif config.experiment.method == Method.z_test_linreg:

            slopes = []
            slopes_std = []
            num_subs = []

            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)

                item_id = config_child.advanced_dict[item]
                slopes.append(config_child.advanced_data['slope'][item_id])
                slopes_std.append(config_child.advanced_data['slope_std'][item_id])
                num_subs.append(len(config_child.attributes_dict['age']))

            z_test_slope_proc(slopes, slopes_std, num_subs, config.metrics)

        elif config.experiment.method == Method.ancova:

            x_all = []
            y_all = []
            category_all = []
            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)

                x = self.get_strategy.get_target(config_child)
                y = self.get_strategy.get_single_base(config_child, [item])[0]
                x_all += x
                y_all += list(y)
                category_all += [list(string.ascii_lowercase)[configs_child.index(config_child)]] * len(x)

            data = {'x': x_all, 'y': y_all, 'category': category_all}
            df = pd.DataFrame(data)
            formula = 'y ~ x * category'
            lm = ols(formula, df)
            results = lm.fit()
            p_value = results.pvalues[3]

            config.metrics['p_value'].append(p_value)

        elif config.experiment.method == Method.aggregator:

            metrics_keys = get_method_metrics_keys(config)
            for config_child in configs_child:
                update_parent_dict_with_children(metrics_keys, item, config, config_child)

        elif config.experiment.method == Method.variance:

            targets = self.get_strategy.get_target(config)
            data = self.get_strategy.get_single_base(config, [item])
            targets = np.squeeze(np.asarray(targets))
            data = np.squeeze(np.asarray(data))

            semi_window = config.experiment.method_params['semi_window']
            box_b = config.experiment.method_params['box_b']
            box_t = config.experiment.method_params['box_t']

            process_variance(targets, data, semi_window, box_b, box_t, config.metrics)

            xs = get_box_xs(targets)
            ys_b, ys_t = fit_variance(xs, config.metrics)

            diff_begin = abs(ys_t[0] - ys_b[0])
            diff_end = abs(ys_t[-1] - ys_b[-1])

            config.metrics['increasing_div'].append(max(diff_begin, diff_end) / min(diff_begin, diff_end))
            config.metrics['increasing_sub'].append(abs(diff_begin - diff_end))

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
                                      DataType.residuals_common,
                                      DataType.residuals_special]:
            self.iterate(config, configs_child)

        elif config.experiment.data == DataType.epimutations:

            item = 'epimutations'

            if config.experiment.method == Method.linreg:

                targets = self.get_strategy.get_target(config)
                indexes = config.attributes_indexes
                x = sm.add_constant(targets)
                y = np.zeros(len(indexes), dtype=int)

                for subj_id in range(0, len(indexes)):
                    col_id = indexes[subj_id]

                    subj_col = self.get_strategy.get_single_base(config, [col_id])
                    y[subj_id] = np.sum(subj_col)

                y = np.log(y)

                process_linreg(x, y, config.metrics)

            elif config.experiment.method == Method.ancova:

                x_all = []
                y_all = []
                category_all = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)
                    x = self.get_strategy.get_target(config_child)
                    indexes = config_child.attributes_indexes
                    y = np.zeros(len(indexes), dtype=int)
                    for subj_id in range(0, len(indexes)):
                        col_id = indexes[subj_id]
                        subj_col = self.get_strategy.get_single_base(config_child, [col_id])
                        y[subj_id] = np.sum(subj_col)
                    y = np.log(y)

                    x_all += x
                    y_all += list(y)
                    category_all += [list(string.ascii_lowercase)[configs_child.index(config_child)]] * len(x)

                data = {'x': x_all, 'y': y_all, 'category': category_all}
                df = pd.DataFrame(data)
                formula = 'y ~ x * category'
                lm = ols(formula, df)
                results = lm.fit()
                p_value = results.pvalues[3]

                config.metrics['p_value'].append(p_value)

            elif config.experiment.method == Method.z_test_linreg:

                slopes = []
                slopes_std = []
                num_subs = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    item_id = config_child.advanced_dict[item]
                    slopes.append(config_child.advanced_data['slope'][item_id])
                    slopes_std.append(config_child.advanced_data['slope_std'][item_id])
                    num_subs.append(len(config_child.attributes_dict['age']))

                z_test_slope_proc(slopes, slopes_std, num_subs, config.metrics)

            elif config.experiment.method == Method.polygon:

                xs = []
                ys = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    targets = self.get_strategy.get_target(config_child)
                    x = targets
                    indexes = config_child.attributes_indexes
                    y = np.zeros(len(indexes), dtype=int)
                    for subj_id in range(0, len(indexes)):
                        col_id = indexes[subj_id]

                        subj_col = self.get_strategy.get_single_base(config_child, [col_id])
                        y[subj_id] = np.sum(subj_col)
                    y = np.log(y)

                    xs.append(x)
                    ys.append(y)

                if config.experiment.method_params['method'] == Method.linreg:
                    process_linreg_polygon(configs_child, item, xs, config.metrics)

                elif config.experiment.method_params['method'] == Method.variance:
                    process_variance_polygon(configs_child, item, xs, config.metrics)

            elif config.experiment.method == Method.aggregator:

                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

            config.metrics['item'].append(item)
            config.metrics['aux'].append('')

        elif config.experiment.data == DataType.entropy:

            item = 'entropy'

            if config.experiment.method == Method.linreg:

                indexes = config.attributes_indexes

                targets = self.get_strategy.get_target(config)
                x = sm.add_constant(targets)
                y = self.get_strategy.get_single_base(config, indexes)

                process_linreg(x, y, config.metrics)

            elif config.experiment.method == Method.ancova:

                x_all = []
                y_all = []
                category_all = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    indexes = config_child.attributes_indexes
                    x = self.get_strategy.get_target(config_child)
                    y = self.get_strategy.get_single_base(config_child, indexes)
                    x_all += x
                    y_all += list(y)
                    category_all += [list(string.ascii_lowercase)[configs_child.index(config_child)]] * len(x)

                data = {'x': x_all, 'y': y_all, 'category': category_all}
                df = pd.DataFrame(data)
                formula = 'y ~ x * category'
                lm = ols(formula, df)
                results = lm.fit()
                p_value = results.pvalues[3]

                config.metrics['p_value'].append(p_value)

            elif config.experiment.method == Method.z_test_linreg:

                slopes = []
                slopes_std = []
                num_subs = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    item_id = config_child.advanced_dict[item]
                    slopes.append(config_child.advanced_data['slope'][item_id])
                    slopes_std.append(config_child.advanced_data['slope_std'][item_id])
                    num_subs.append(len(config_child.attributes_dict['age']))

                z_test_slope_proc(slopes, slopes_std, num_subs, config.metrics)

            elif config.experiment.method == Method.polygon:

                xs = []
                ys = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    indexes = config_child.attributes_indexes
                    x = self.get_strategy.get_target(config_child)
                    y = self.get_strategy.get_single_base(config_child, indexes)
                    xs.append(x)
                    ys.append(y)

                if config.experiment.method_params['method'] == Method.linreg:
                    process_linreg_polygon(configs_child, item, xs, config.metrics)

                elif config.experiment.method_params['method'] == Method.variance:
                    process_variance_polygon(configs_child, item, xs, config.metrics)

            elif config.experiment.method == Method.aggregator:

                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

            config.metrics['item'].append(item)
            config.metrics['aux'].append('')

        elif config.experiment.data == DataType.cells:

            cells = config.attributes.cells
            cells_types = cells.types
            item = str(cells_types)

            if config.experiment.method == Method.linreg:

                targets = self.get_strategy.get_target(config)
                x = sm.add_constant(targets)

                if isinstance(cells_types, list):
                    y = np.zeros(len(x))
                    num_cell_types = 0
                    for cell_type in cells_types:
                        if cell_type in config.cells_dict:
                            y += np.asarray(config.cells_dict[cell_type])
                            num_cell_types += 1
                    y /= num_cell_types
                else:
                    y = config.cells_dict[cells_types]

                process_linreg(x, y, config.metrics)

            elif config.experiment.method == Method.ancova:

                x_all = []
                y_all = []
                category_all = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    x = self.get_strategy.get_target(config_child)

                    cells = config_child.attributes.cells
                    cells_types = cells.types
                    if isinstance(cells_types, list):
                        y = np.zeros(len(x))
                        num_cell_types = 0
                        for cell_type in cells_types:
                            if cell_type in config_child.cells_dict:
                                y += np.asarray(config_child.cells_dict[cell_type])
                                num_cell_types += 1
                        y /= num_cell_types
                    else:
                        y = config_child.cells_dict[cells_types]

                    x_all += x
                    y_all += list(y)
                    category_all += [list(string.ascii_lowercase)[configs_child.index(config_child)]] * len(x)

                data = {'x': x_all, 'y': y_all, 'category': category_all}
                df = pd.DataFrame(data)
                formula = 'y ~ x * category'
                lm = ols(formula, df)
                results = lm.fit()
                p_value = results.pvalues[3]

                config.metrics['p_value'].append(p_value)

            elif config.experiment.method == Method.z_test_linreg:

                slopes = []
                slopes_std = []
                num_subs = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    item_id = config_child.advanced_dict[item]
                    slopes.append(config_child.advanced_data['slope'][item_id])
                    slopes_std.append(config_child.advanced_data['slope_std'][item_id])
                    num_subs.append(len(config_child.attributes_dict['age']))

                z_test_slope_proc(slopes, slopes_std, num_subs, config.metrics)

            elif config.experiment.method == Method.polygon:

                xs = []
                ys = []
                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

                    x = self.get_strategy.get_target(config_child)

                    cells = config_child.attributes.cells
                    cells_types = cells.types
                    if isinstance(cells_types, list):
                        y = np.zeros(len(x))
                        num_cell_types = 0
                        for cell_type in cells_types:
                            if cell_type in config_child.cells_dict:
                                y += np.asarray(config_child.cells_dict[cell_type])
                                num_cell_types += 1
                        y /= num_cell_types
                    else:
                        y = config_child.cells_dict[cells_types]

                    xs.append(x)
                    ys.append(list(y))

                if config.experiment.method_params['method'] == Method.linreg:
                    process_linreg_polygon(configs_child, item, xs, config.metrics)

                elif config.experiment.method_params['method'] == Method.variance:
                    process_variance_polygon(configs_child, item, xs, config.metrics)

            elif config.experiment.method == Method.aggregator:

                metrics_keys = get_method_metrics_keys(config)
                for config_child in configs_child:
                    update_parent_dict_with_children(metrics_keys, item, config, config_child)

            config.metrics['item'].append(item)
            config.metrics['aux'].append('')


class ClockRunStrategy(RunStrategy):

    def single(self, item, config, configs_child):
        pass

    def iterate(self, config, configs_child):
        pass

    def run(self, config, configs_child):

        if config.experiment.data in [DataType.betas, DataType.betas_adj, DataType.residuals_common,
                                      DataType.residuals_special]:

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

                        build_clock_linreg(clock)

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

                        build_clock_linreg(clock)

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

                    build_clock_linreg(clock)


class PlotRunStrategy(RunStrategy):

    def single(self, item, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals_common,
                                      DataType.residuals_special]:

            if config.experiment.method == Method.scatter:

                line = config.experiment.method_params['line']
                add = config.experiment.method_params['add']
                fit = config.experiment.method_params['fit']
                semi_window = config.experiment.method_params['semi_window']

                plot_data = []
                num_points = []
                for config_child in configs_child:

                    curr_plot_data = []

                    # Plot data
                    targets = self.get_strategy.get_target(config_child)
                    num_points.append(len(targets))
                    data = self.get_strategy.get_single_base(config_child, [item])[0]

                    # Colors setup
                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]
                    coordinates = color[4:-1].split(',')
                    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.1) + ')'
                    color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

                    # Adding scatter
                    scatter = go.Scatter(
                        x=targets,
                        y=data,
                        name=get_names(config_child, config.experiment.method_params),
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
                            params={
                                'intercept': [intercept],
                                'slope': [slope],
                                'intercept_std': [intercept_std],
                                'slope_std': [slope_std]
                            },
                            method=config_child.experiment.method
                        )
                        scatter = pr.get_scatter(color_transparent)
                        curr_plot_data.append(scatter)

                    # Adding box curve
                    if fit == 'no' and semi_window != 'none':
                        box_b = config.experiment.method_params['box_b']
                        box_t = config.experiment.method_params['box_t']
                        xs, bs, ms, ts = process_box(targets, data, semi_window, box_b, box_t)

                        scatter = go.Scatter(
                            x=xs,
                            y=bs,
                            name=get_names(config_child, config.experiment.method_params),
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
                            name=get_names(config_child, config.experiment.method_params),
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
                            name=get_names(config_child, config.experiment.method_params),
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
                        box_b = config.experiment.method_params['box_b']
                        box_t = config.experiment.method_params['box_t']

                        metrics_dict = {}
                        init_variance_metrics_dict(metrics_dict, 'box_b')
                        init_variance_metrics_dict(metrics_dict, 'box_m')
                        init_variance_metrics_dict(metrics_dict, 'box_t')

                        xs, _, _, _ = process_variance(targets, data, semi_window, box_b, box_t, metrics_dict)

                        ys_b, ys_t = fit_variance(xs, metrics_dict)

                        scatter = go.Scatter(
                            x=xs,
                            y=ys_t,
                            name=get_names(config_child, config.experiment.method_params),
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
                            name=get_names(config_child, config.experiment.method_params),
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
                config.experiment_data['data'].append(curr_data)

            elif config.experiment.method == Method.variance_histogram:

                plot_data = {
                    'hist_data': [],
                    'group_labels': [],
                    'colors': []
                }

                for config_child in configs_child:

                    plot_data['group_labels'].append(str(config_child.attributes.observables))
                    plot_data['colors'].append(cl.scales['8']['qual']['Set1'][configs_child.index(config_child)])

                    targets = self.get_strategy.get_target(config_child)
                    data = self.get_strategy.get_single_base(config_child, [item])[0]

                    if config_child.experiment.method == Method.linreg:
                        x = sm.add_constant(targets)
                        y = data

                        results = sm.OLS(y, x).fit()

                        plot_data['hist_data'].append(results.resid)

                config.experiment_data['data'].append(plot_data)

    def iterate(self, config, configs_child):
        items = config.experiment.method_params['items']
        for item in items:
            if item in config.base_dict:
                print(item)
                config.experiment_data['item'].append(item)
                self.single(item, config, configs_child)

    def run(self, config, configs_child):

        if config.experiment.data in [DataType.betas,
                                      DataType.betas_adj,
                                      DataType.residuals_common,
                                      DataType.residuals_special]:

            if config.experiment.method in [Method.scatter, Method.variance_histogram]:
                self.iterate(config, configs_child)

            elif config.experiment.method == Method.scatter_comparison:
                pass

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

        elif config.experiment.data == DataType.epimutations:

            if config.experiment.method == Method.scatter:

                plot_data = []
                num_points = []

                y_type = config.experiment.method_params['y_type']

                for config_child in configs_child:

                    curr_plot_data = []

                    indexes = config_child.attributes_indexes
                    num_points.append(len(indexes))

                    x = self.get_strategy.get_target(config_child)
                    y = np.zeros(len(indexes), dtype=int)

                    for subj_id in range(0, len(indexes)):
                        col_id = indexes[subj_id]
                        subj_col = self.get_strategy.get_single_base(config_child, [col_id])
                        y[subj_id] = np.sum(subj_col)

                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]
                    coordinates = color[4:-1].split(',')
                    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.7) + ')'
                    color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

                    scatter = go.Scatter(
                        x=x,
                        y=y,
                        name=get_names(config_child, config.experiment.method_params),
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=color_transparent,
                            line=dict(
                                width=1,
                                color=color_border,
                            )
                        ),
                    )
                    curr_plot_data.append(scatter)

                    # Adding regression line

                    x_linreg = sm.add_constant(x)
                    if y_type == 'log':
                        y_linreg = np.log(y)
                    else:
                        y_linreg = y

                    results = sm.OLS(y_linreg, x_linreg).fit()

                    intercept = results.params[0]
                    slope = results.params[1]

                    x_min = np.min(x)
                    x_max = np.max(x)
                    if y_type == 'log':
                        y_min = np.exp(slope * x_min + intercept)
                        y_max = np.exp(slope * x_max + intercept)
                    else:
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

                    plot_data.append(curr_plot_data)

                order = np.argsort(num_points)[::-1]
                config.experiment_data['data'] = []
                for index in order:
                    config.experiment_data['data'] += plot_data[index]

            elif config.experiment.method == Method.range:

                plot_data = []

                borders = config.experiment.method_params['borders']

                for config_child in configs_child:

                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]
                    coordinates = color[4:-1].split(',')
                    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.5) + ')'

                    indexes = config_child.attributes_indexes

                    x = self.get_strategy.get_target(config_child)
                    y = np.zeros(len(indexes), dtype=int)

                    for subj_id in range(0, len(indexes)):
                        col_id = indexes[subj_id]
                        subj_col = self.get_strategy.get_single_base(config_child, [col_id])
                        y[subj_id] = np.sum(subj_col)

                    for seg_id in range(0, len(borders) - 1):
                        x_center = (borders[seg_id + 1] + borders[seg_id]) * 0.5
                        curr_box = []
                        for subj_id in range(0, len(indexes)):
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

                config.experiment_data['data'] = plot_data

        elif config.experiment.data == DataType.entropy:

            if config.experiment.method == Method.scatter:

                plot_data = []
                num_points = []

                for config_child in configs_child:
                    curr_plot_data = []
                    indexes = config_child.attributes_indexes
                    num_points.append(len(indexes))

                    x = self.get_strategy.get_target(config_child)
                    y = self.get_strategy.get_single_base(config_child, indexes)

                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]
                    coordinates = color[4:-1].split(',')
                    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.7) + ')'
                    color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

                    scatter = go.Scatter(
                        x=x,
                        y=y,
                        name=get_names(config_child, config.experiment.method_params),
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=color_transparent,
                            line=dict(
                                width=1,
                                color=color_border,
                            )
                        ),
                    )
                    curr_plot_data.append(scatter)

                    # Adding regression line

                    x_linreg = sm.add_constant(x)
                    y_linreg = y

                    results = sm.OLS(y_linreg, x_linreg).fit()

                    intercept = results.params[0]
                    slope = results.params[1]

                    x_min = np.min(x)
                    x_max = np.max(x)
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

                    plot_data.append(curr_plot_data)

                order = np.argsort(num_points)[::-1]
                config.experiment_data['data'] = []
                for index in order:
                    config.experiment_data['data'] += plot_data[index]

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

        elif config.experiment.data == DataType.cells:

            if config.experiment.method == Method.scatter:

                plot_data = []
                num_points = []

                for config_child in configs_child:
                    curr_plot_data = []
                    indexes = config_child.attributes_indexes
                    num_points.append(len(indexes))

                    x = self.get_strategy.get_target(config_child)
                    cells = config_child.attributes.cells
                    cells_types = cells.types
                    if isinstance(cells_types, list):
                        y = np.zeros(len(x))
                        num_cell_types = 0
                        for cell_type in cells_types:
                            if cell_type in config_child.cells_dict:
                                y += np.asarray(config_child.cells_dict[cell_type])
                                num_cell_types += 1
                        y /= num_cell_types
                    else:
                        y = config_child.cells_dict[cells_types]

                    color = cl.scales['8']['qual']['Set1'][configs_child.index(config_child)]
                    coordinates = color[4:-1].split(',')
                    color_transparent = 'rgba(' + ','.join(coordinates) + ',' + str(0.7) + ')'
                    color_border = 'rgba(' + ','.join(coordinates) + ',' + str(0.8) + ')'

                    scatter = go.Scatter(
                        x=x,
                        y=y,
                        name=get_names(config_child, config.experiment.method_params),
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=color_transparent,
                            line=dict(
                                width=1,
                                color=color_border,
                            )
                        ),
                    )
                    curr_plot_data.append(scatter)

                    # Adding regression line

                    x_linreg = sm.add_constant(x)
                    y_linreg = y

                    results = sm.OLS(y_linreg, x_linreg).fit()

                    intercept = results.params[0]
                    slope = results.params[1]

                    x_min = np.min(x)
                    x_max = np.max(x)
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

                    plot_data.append(curr_plot_data)

                order = np.argsort(num_points)[::-1]
                config.experiment_data['data'] = []
                for index in order:
                    config.experiment_data['data'] += plot_data[index]


class CreateRunStrategy(RunStrategy):

    def single(self, item, config_child, configs_child):
        pass

    def iterate(self, config, configs_child):
        pass

    def run(self, config, configs_child):
        pass
