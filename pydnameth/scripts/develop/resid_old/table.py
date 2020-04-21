from pydnameth.config.experiment.types import DataType, Method
from pydnameth.scripts.develop.table import table


def resid_old_table_linreg(
    data,
    annotations,
    attributes,
    data_params,
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.resid_old,
        method=Method.linreg,
        data_params=data_params,
        task_params=None,
        method_params=None
    )
