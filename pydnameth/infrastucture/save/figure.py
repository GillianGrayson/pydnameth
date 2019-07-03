import plotly


def save_figure(fn, fig):
    # plotly.offline.plot(fig, filename=fn + '.html', auto_open=False, show_link=True)
    plotly.io.write_image(fig, fn + '.png')
    plotly.io.write_image(fig, fn + '.pdf')
