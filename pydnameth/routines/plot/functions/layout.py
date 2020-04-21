import plotly.graph_objs as go
from pydnameth.config.experiment.types import DataType, Method
from pydnameth.routines.common import get_axis, get_legend, get_margin


def get_layout(config, title_text=''):
    layout = None

    if config.experiment.data in [DataType.betas,
                                  DataType.betas_adj,
                                  DataType.residuals,
                                  DataType.resid_old,
                                  DataType.epimutations,
                                  DataType.entropy,
                                  DataType.cells,
                                  DataType.genes]:

        y_title = 'Methylation level'
        if config.experiment.data in [DataType.residuals, DataType.resid_old]:
            y_title = 'Residuals'
        if config.experiment.data == DataType.epimutations:
            y_title = 'log(SEM number)'
        elif config.experiment.data == DataType.entropy:
            y_title = 'Entropy'
        elif config.experiment.data == DataType.cells:
            y_title = 'Cells'

        if config.experiment.method in [Method.scatter, Method.variance_histogram]:

            layout = go.Layout(
                title=dict(
                    text=title_text,
                    font=dict(
                        family='Arial',
                        size=33,
                    )
                ),
                autosize=True,
                margin=go.layout.Margin(
                    l=110,
                    r=10,
                    b=80,
                    t=85,
                    pad=0
                ),
                barmode='overlay',
                legend=dict(
                    font=dict(
                        family='Arial',
                        size=16,
                    ),
                    orientation="h",
                    x=0.33,
                    y=1.11,
                ),
                xaxis=get_axis(config.attributes.target.capitalize()),
                yaxis=get_axis(y_title),
            )

        elif config.experiment.method == Method.range:

            layout = go.Layout(
                title=dict(
                    font=dict(
                        family='Arial',
                        size=33,
                    )
                ),
                autosize=True,
                margin=go.layout.Margin(
                    l=120,
                    r=10,
                    b=70,
                    t=10,
                    pad=0
                ),
                barmode='overlay',
                showlegend=False,
                xaxis=dict(
                    title=config.attributes.target.capitalize(),
                    showgrid=True,
                    showline=True,
                    mirror='ticks',
                    titlefont=dict(
                        family='Arial',
                        size=33,
                        color='black'
                    ),
                    showticklabels=True,
                    tickangle=0,
                    tickfont=dict(
                        family='Arial',
                        size=18,
                        color='black'
                    ),
                    exponentformat='e',
                    showexponent='all'
                ),
                yaxis=get_axis(y_title),
            )

        elif config.experiment.method in [Method.curve]:

            x_title = config.experiment.method_params['x']
            y_title = config.experiment.method_params['y']

            layout = go.Layout(
                autosize=True,
                margin=go.layout.Margin(
                    l=95,
                    r=10,
                    b=80,
                    t=10,
                    pad=0
                ),
                barmode='overlay',
                legend=dict(
                    font=dict(
                        family='Arial',
                        size=16,
                    ),
                    orientation="h",
                    x=0.33,
                    y=1.11,
                ),
                xaxis=dict(
                    title=x_title,
                    zeroline=False,
                    showgrid=True,
                    showline=True,
                    mirror='ticks',
                    titlefont=dict(
                        family='Arial',
                        size=33,
                        color='black'
                    ),
                    showticklabels=True,
                    tickangle=0,
                    tickfont=dict(
                        family='Arial',
                        size=30,
                        color='black'
                    ),
                    exponentformat='e',
                    showexponent='all'
                ),
                yaxis=get_axis(y_title),
            )

    elif config.experiment.data == DataType.observables:

        if config.experiment.method == Method.histogram:
            layout = go.Layout(
                autosize=True,
                margin=get_margin(),
                barmode=config.experiment.method_params['barmode'],
                legend=get_legend(),
                xaxis=get_axis(config.attributes.target.capitalize()),
                yaxis=get_axis('Count'),
            )

    layout.plot_bgcolor = 'rgba(233,233,233,0)'

    return layout
