import copy
from anytree import Node
from pydnameth.config.config import Config
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.config.experiment.types import DataType, Method, Task
from pydnameth.model.tree import build_tree, calc_tree
from pydnameth.scripts.develop.table import table_aggregator_linreg, table_aggregator_variance, table


def betas_adj_table_aggregator_linreg(
    data,
    annotations,
    attributes,
    observables_list,
    data_params,
):
    table_aggregator_linreg(
        DataType.betas_adj,
        data,
        annotations,
        attributes,
        observables_list,
        data_params=data_params,
    )


def betas_adj_table_aggregator_variance(
    data,
    annotations,
    attributes,
    observables_list,
    data_params,
):
    table_aggregator_variance(
        DataType.betas_adj,
        data,
        annotations,
        attributes,
        observables_list,
        data_params=data_params,
    )


def betas_adj_table_oma(
    data,
    annotations,
    attributes,
    data_params,
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.betas_adj,
        method=Method.oma,
        data_params=data_params,
        task_params=None,
        method_params=None
    )


def betas_adj_table_approach_3(
    data,
    annotations,
    attributes,
    target_list,
    data_params_list
):
    config_root = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.betas_adj,
            task=Task.table,
            method=Method.aggregator,
            data_params=copy.deepcopy(data_params_list[0])
        ),
        annotations=copy.deepcopy(annotations),
        attributes=copy.deepcopy(attributes),
        is_run=True,
        is_root=True
    )
    root = Node(name=str(config_root), config=config_root)

    for id, target in enumerate(target_list):
        attributes_common = copy.deepcopy(attributes)
        attributes_common.target = target
        config_common = Config(
            data=copy.deepcopy(data),
            experiment=Experiment(
                data=DataType.betas_adj,
                task=Task.table,
                method=Method.oma,
                data_params=copy.deepcopy(data_params_list[id])
            ),
            annotations=copy.deepcopy(annotations),
            attributes=attributes_common,
            is_run=True,
            is_root=False
        )
        Node(name=str(config_common), config=config_common, parent=root)

    build_tree(root)
    calc_tree(root)
