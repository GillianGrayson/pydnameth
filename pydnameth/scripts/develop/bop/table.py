from pydnameth.config.experiment.types import DataType, Method
from pydnameth.scripts.develop.table import table


def bop_table_manova(
    data,
    annotations,
    attributes,
    data_params,
    method_params
):
    table(
        data=data,
        annotations=annotations,
        attributes=attributes,
        data_type=DataType.bop,
        method=Method.manova,
        data_params=data_params,
        task_params=None,
        method_params=method_params
    )
