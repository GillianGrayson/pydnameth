from pydnameth.scripts.develop.table import table_z_test_linreg
from pydnameth.config.experiment.types import DataType


def epimutations_table_z_test_linreg(
    data,
    annotations,
    attributes,
    observables_list
):
    table_z_test_linreg(
        data_type=DataType.epimutations,
        data=data,
        annotations=annotations,
        attributes=attributes,
        observables_list=observables_list,
        data_params=None,
        task_params=None,
        method_params=None
    )
