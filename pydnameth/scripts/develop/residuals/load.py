import copy
from pydnameth.config.config import Config
from pydnameth.config.experiment.types import Task
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.model.context import Context
from pydnameth.config.experiment.types import Method, DataType


def load_residuals_config(
    data,
    annotations,
    attributes,
    data_params=None
):
    config = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.residuals,
            task=Task.load,
            method=Method.mock,
            data_params=copy.deepcopy(data_params)
        ),
        annotations=copy.deepcopy(annotations),
        attributes=copy.deepcopy(attributes),
        is_run=True,
        is_root=True
    )

    context = Context(config)
    context.load(config)

    return config
