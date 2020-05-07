import copy
from anytree import Node
from pydnameth.config.config import Config
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.config.experiment.types import DataType, Method, Task
from pydnameth.model.tree import build_tree, calc_tree
from pydnameth.scripts.develop.table import table_aggregator_linreg, table_aggregator_variance, table


def residuals_table_linreg(
    data,
    annotations,
    attributes,
    data_params,
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.residuals,
        method=Method.linreg,
        data_params=data_params,
        task_params=None,
        method_params=None
    )


def residuals_table_aggregator_linreg(
    data,
    annotations,
    attributes,
    observables_list,
    data_params,
):
    table_aggregator_linreg(
        DataType.residuals,
        data,
        annotations,
        attributes,
        observables_list,
        data_params=data_params,
    )


def residuals_table_aggregator_variance(
    data,
    annotations,
    attributes,
    observables_list,
    data_params,
):
    table_aggregator_variance(
        DataType.residuals,
        data,
        annotations,
        attributes,
        observables_list,
        data_params=data_params,
    )


def residuals_table_oma(
    data,
    annotations,
    attributes,
    data_params,
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.residuals,
        method=Method.oma,
        data_params=data_params,
        task_params=None,
        method_params=None
    )


def residuals_table_pbc(
    data,
    annotations,
    attributes,
    data_params,
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.residuals,
        method=Method.pbc,
        data_params=data_params,
        task_params=None,
        method_params=None
    )


def residuals_table_approach_3(
    data,
    annotations,
    attributes,
    target_sex_specific,
    target_age_related,
    data_params_sex_specific,
    data_params_age_related
):
    config_root = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.residuals,
            task=Task.table,
            method=Method.aggregator,
            data_params=copy.deepcopy(data_params_sex_specific)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=copy.deepcopy(attributes),
        is_run=True,
        is_root=True
    )
    root = Node(name=str(config_root), config=config_root)

    attributes_ss = copy.deepcopy(attributes)
    attributes_ss.target = target_sex_specific
    config_ss = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.residuals,
            task=Task.table,
            method=Method.pbc,
            data_params=copy.deepcopy(data_params_sex_specific)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=attributes_ss,
        is_run=True,
        is_root=False
    )
    Node(name=str(config_ss), config=config_ss, parent=root)

    attributes_ar = copy.deepcopy(attributes)
    attributes_ar.target = target_age_related
    config_ar = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.residuals,
            task=Task.table,
            method=Method.oma,
            data_params=copy.deepcopy(data_params_age_related)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=attributes_ar,
        is_run=True,
        is_root=False
    )
    Node(name=str(config_ar), config=config_ar, parent=root)

    build_tree(root)
    calc_tree(root)
