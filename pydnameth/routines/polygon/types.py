import numpy as np
from shapely import geometry
import plotly.graph_objs as go


class PolygonRoutines:
    def __init__(self,
                 x,
                 metrics_dict,
                 suffix
                 ):
        self.x = x
        self.metrics_dict = metrics_dict
        self.suffix = suffix

    def get_border_points(self):

        sigma = 3.0

        intercept = self.metrics_dict['intercept' + self.suffix][0]
        slope = self.metrics_dict['slope' + self.suffix][0]
        intercept_std = self.metrics_dict['intercept_std' + self.suffix][0]
        slope_std = self.metrics_dict['slope_std' + self.suffix][0]

        x_min = np.min(self.x)
        x_max = np.max(self.x)
        y_min = slope * x_min + intercept
        y_max = slope * x_max + intercept
        slope_tmp = slope + sigma * slope_std
        y_tmp = slope_tmp * x_max + intercept
        y_diff = sigma * np.abs(intercept_std) + np.abs(y_tmp - y_max)
        y_min_up = y_min + y_diff
        y_min_down = y_min - y_diff
        y_max_up = y_max + y_diff
        y_max_down = y_max - y_diff

        points = [
            geometry.Point(x_min, y_min_down),
            geometry.Point(x_max, y_max_down),
            geometry.Point(x_max, y_max_up),
            geometry.Point(x_min, y_min_up),
        ]

        return points

    def get_polygon(self):
        points = self.get_border_points()
        polygon = geometry.Polygon([[point.x, point.y] for point in points])
        return polygon

    def get_scatter(self, color):
        points = self.get_border_points()
        xs = [point.x for point in points] + [points[0].x]
        ys = [point.y for point in points] + [points[0].y]
        scatter = go.Scatter(
            x=xs,
            y=ys,
            fill='tozerox',
            mode='lines',
            marker=dict(
                opacity=0.75,
                color=color,
                line=dict(width=8)
            ),
            showlegend=False
        )
        return scatter
