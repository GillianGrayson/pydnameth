import copy
from pydnameth.config.config import Config
from pydnameth.config.experiment.types import DataType
from pydnameth.config.experiment.experiment import Experiment
from pydnameth.config.attributes.attributes import Attributes
from pydnameth.config.annotations.annotations import Annotations
from pydnameth.infrastucture.load.epimutations import load_epimutations


def epimutations_load(data):

    config = Config(
        data=copy.deepcopy(data),
        experiment=Experiment(
            data=DataType.epimutations,
            task=None,
            method=None
        ),
        annotations=Annotations(
            name='annotations',
            type='450k',
            exclude='bad_cpgs',
            select_dict={
                'CHR': ['-X', '-Y']
            }
        ),
        attributes=Attributes(
            target=None,
            observables=None,
            cells=None
        ),
        is_run=True,
        is_root=True
    )

    load_epimutations(config)
