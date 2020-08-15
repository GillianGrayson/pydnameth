import copy
from anytree import Node
from pydnameth.config.config import Config
from pydnameth.config.experiment.types import DataType, Task, Method
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.config.attributes.attributes import Observables, Cells, Attributes
from pydnameth.config.annotations.annotations import Annotations
from pydnameth.model.tree import build_tree, calc_tree


def betas_horvath_calculator_create_regular(
    data,
    data_params
):
    annotations = Annotations(
        name='annotations',
        type='450k',
        exclude='none',
        select_dict={}
    )

    cells = Cells(
        name='cells',
        types='any'
    )

    observables = Observables(
        name='observables_part(control)',
        types={}
    )

    attributes = Attributes(
        target='age',
        observables=observables,
        cells=cells
    )

    config_root = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.betas_horvath_calculator,
            task=Task.create,
            method=Method.regular,
            data_params=copy.deepcopy(data_params)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=copy.deepcopy(attributes),
        is_run=True,
        is_root=True
    )

    root = Node(name=str(config_root), config=config_root)
    build_tree(root)
    calc_tree(root)
