import abc
from pydnameth.infrastucture.save.figure import save_figure
from pydnameth.infrastucture.save.table import save_table_dict
from pydnameth.infrastucture.path import get_save_path
from pydnameth.infrastucture.file_name import get_file_name
import glob
from pathlib import Path
from pydnameth.config.experiment.types import Method


class SaveStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def save(self, config, configs_child):
        pass

    @abc.abstractmethod
    def is_result_exist(self, config, configs_child):
        pass


class TableSaveStrategy(SaveStrategy):

    def save(self, config, configs_child):
        fn = get_save_path(config) + '/' + get_file_name(config)
        save_table_dict(fn, config.metrics)

    def is_result_exist(self, config, configs_child):
        fn = get_save_path(config) + '/' + \
            get_file_name(config) + '.*'
        if glob.glob(fn):
            return True
        else:
            return False


class ClockSaveStrategy(SaveStrategy):

    def save(self, config, configs_child):
        fn = get_save_path(config) + '/' + \
            get_file_name(config)
        save_table_dict(fn, config.metrics)

    def is_result_exist(self, config, configs_child):
        fn = get_save_path(config) + '/' + \
            get_file_name(config) + '.*'
        if glob.glob(fn):
            return True
        else:
            return False


class PlotSaveStrategy(SaveStrategy):

    def save(self, config, configs_child):

        if isinstance(config.experiment_data['fig'], list):

            for fig_id, fig in enumerate(config.experiment_data['fig']):

                if config.experiment.method == Method.scatter:
                    item = config.experiment_data['item'][fig_id]
                    config.experiment.method_params.pop('items', None)
                    config.experiment.method_params.pop('x_ranges', None)
                    config.experiment.method_params.pop('y_ranges', None)
                    config.experiment.method_params['item'] = item

                fn = get_save_path(config) + '/' + get_file_name(config)
                save_figure(fn, fig)
        else:
            fn = get_save_path(config) + '/' + get_file_name(config)
            save_figure(fn, config.experiment_data['fig'])

    def is_result_exist(self, config, configs_child):
        fn = get_save_path(config) + '/' + \
            get_file_name(config) + '.pdf'
        if Path(fn).is_file():
            return True
        else:
            return False


class CreateSaveStrategy(SaveStrategy):

    def save(self, config, configs_child):
        pass

    def is_result_exist(self, config, configs_child):
        pass
