import abc
from pydnameth.config.experiment.types import Task
from pydnameth.infrastucture.load.betas import load_betas
from pydnameth.infrastucture.load.betas_adj import load_betas_adj
from pydnameth.infrastucture.load.betas_horvath_calculator import load_betas_horvath_calculator
from pydnameth.infrastucture.load.betas_spec import load_betas_spec
from pydnameth.infrastucture.load.residuals import load_residuals_common
from pydnameth.infrastucture.load.table import load_table_dict
from pydnameth.infrastucture.load.epimutations import load_epimutations
from pydnameth.infrastucture.load.entropy import load_entropy
from pydnameth.infrastucture.load.cells import load_cells
from pydnameth.infrastucture.load.genes import load_genes


class LoadStrategy(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def load(self, config, configs_child):
        pass

    def inherit_childs(self, config, configs_child):
        for config_child in configs_child:
            config_child.base_list = config.base_list
            config_child.base_dict = config.base_dict
            config_child.base_missed_dict = config.base_missed_dict
            config_child.base_data = config.base_data

    def load_child(self, config_child):

        if config_child.experiment.task in [Task.table, Task.clock]:

            config_child.advanced_data = load_table_dict(config_child)
            config_child.advanced_list = config_child.base_list
            config_child.advanced_dict = {}
            row_id = 0
            for item in config_child.advanced_data['item']:
                config_child.advanced_dict[item] = row_id
                row_id += 1

        elif config_child.experiment.task in [Task.plot]:
            config_child.advanced_data = config_child.experiment_data


class BetasLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_betas(config)
            config.base_list = config.cpg_list
            config.base_dict = config.betas_dict
            config.base_missed_dict = config.betas_missed_dict
            config.base_data = config.betas_data

            self.inherit_childs(config, configs_child)

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)


class BetasAdjLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_betas_adj(config)
            config.base_list = config.cpg_list
            config.base_dict = config.betas_adj_dict
            config.base_data = config.betas_adj_data
            config.base_missed_dict = config.betas_adj_missed_dict

            self.inherit_childs(config, configs_child)

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)


class BetasHorvathCalculatorLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        load_betas_horvath_calculator(config)


class BetasSpecLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        load_betas_spec(config)


class ResidualsLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_residuals_common(config)
            config.base_list = config.cpg_list
            config.base_dict = config.residuals_dict
            config.base_data = config.residuals_data
            config.base_missed_dict = config.residuals_missed_dict

            self.inherit_childs(config, configs_child)

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)


class GenesLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_genes(config)
            config.base_list = config.genes_list
            config.base_dict = config.genes_dict
            config.base_data = config.genes_data
            config.base_missed_dict = config.genes_missed_dict

            self.inherit_childs(config, configs_child)

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)


class EpimutationsLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_epimutations(config)
            config.base_list = config.epimutations_list
            config.base_dict = config.epimutations_dict
            config.base_data = config.epimutations_data
            config.base_missed_dict = config.epimutations_missed_dict

            self.inherit_childs(config, configs_child)

            for config_child in configs_child:
                config_child.betas_dict = config.betas_dict

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)


class EntropyLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_entropy(config)
            config.base_list = config.entropy_list
            config.base_dict = config.entropy_dict
            config.base_data = config.entropy_data
            config.base_missed_dict = config.entropy_missed_dict

            self.inherit_childs(config, configs_child)

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)


class ObservablesLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        pass


class CellsLoadStrategy(LoadStrategy):

    def load(self, config, configs_child):
        if config.is_init:
            load_cells(config)
            config.base_list = config.cells_list
            config.base_dict = config.cells_dict
            config.base_data = config.cells_dict
            config.base_missed_dict = config.cells_missed_dict

            self.inherit_childs(config, configs_child)

        if config.is_load_child:

            for config_child in configs_child:
                self.load_child(config_child)
