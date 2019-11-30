import copy
from anytree import Node
from pydnameth.config.config import Config
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.config.attributes.attributes import Observables, Cells, Attributes
from pydnameth.config.experiment.types import DataType, Method, Task
from pydnameth.model.tree import build_tree, calc_tree
from pydnameth.scripts.develop.table import table_aggregator_linreg, table_aggregator_variance, table


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


def residuals_table_approach_3(
    data,
    annotations,
    attributes,
    observables_list,
    target_common,
    target_separated,
    data_params_common,
    data_params_separated
):
    config_root = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.residuals,
            task=Task.table,
            method=Method.aggregator,
            data_params=copy.deepcopy(data_params_common)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=copy.deepcopy(attributes),
        is_run=True,
        is_root=True
    )
    root = Node(name=str(config_root), config=config_root)

    attributes_common = copy.deepcopy(attributes)
    attributes_common.target = target_common
    config_common = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.residuals,
            task=Task.table,
            method=Method.oma,
            data_params=copy.deepcopy(data_params_common)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=attributes_common,
        is_run=True,
        is_root=False
    )
    Node(name=str(config_common), config=config_common, parent=root)

    for d in observables_list:
        curr_observables = Observables(
            name=copy.deepcopy(attributes.observables.name),
            types=d
        )

        curr_cells = Cells(
            name=copy.deepcopy(attributes.cells.name),
            types=copy.deepcopy(attributes.cells.types)
        )

        curr_attributes = Attributes(
            target=target_separated,
            observables=curr_observables,
            cells=curr_cells,
        )

        curr_config = Config(
            data=copy.deepcopy(data),
            experiment=Experiment(
                data=DataType.residuals,
                task=Task.table,
                method=Method.oma,
                data_params=copy.deepcopy(data_params_separated),
            ),
            annotations=copy.deepcopy(annotations),
            attributes=curr_attributes,
            is_run=True,
            is_root=False
        )
        Node(name=str(curr_config), config=curr_config, parent=root)

    build_tree(root)
    calc_tree(root)
