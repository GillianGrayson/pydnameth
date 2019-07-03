from pydnameth.scripts.develop.table import table_z_test_linreg, table_ancova
from pydnameth.config.experiment.types import DataType


def cells_table_z_test_linreg(
    data,
    annotations,
    attributes,
    observables_list
):
    table_z_test_linreg(
        data_type=DataType.cells,
        data=data,
        annotations=annotations,
        attributes=attributes,
        observables_list=observables_list,
        data_params=None,
        task_params=None,
        method_params=None
    )


def cells_table_ancova(
    data,
    annotations,
    attributes,
    observables_list
):
    table_ancova(
        data_type=DataType.cells,
        data=data,
        annotations=annotations,
        attributes=attributes,
        observables_list=observables_list,
        data_params=None,
        task_params=None,
        method_params=None
    )
