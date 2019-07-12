import numpy as np
from shapely import geometry
from pydnameth.routines.polygon.types import PolygonRoutines
from pydnameth.routines.variance.functions import fit_variance, get_box_xs
from pydnameth.routines.common import find_nearest_id, dict_slice


def process_linreg_polygon(configs_child, item, xs, metrics_dict):

    polygons_region = []
    polygons_slope = []
    max_abs_slope = 0.0

    mins = [min(x) for x in xs]
    maxs = [max(x) for x in xs]
    border_l = max(mins)
    border_r = min(maxs)
    if border_l > border_r:
        raise ValueError('Polygons borders are not consistent')

    for config_child_id, config_child in enumerate(configs_child):
        targets = xs[config_child_id]
        item_id = config_child.advanced_dict[item]

        local_dict = dict_slice(config_child.advanced_data, item_id)

        slope = config_child.advanced_data['slope'][item_id]
        slope_std = config_child.advanced_data['slope_std'][item_id]

        pr = PolygonRoutines(
            x=targets,
            params=local_dict,
            method=config_child.experiment.method
        )
        points_region = pr.get_border_points()

        points_slope = [
            geometry.Point(slope - 3.0 * slope_std, 0.0),
            geometry.Point(slope + 3.0 * slope_std, 0.0),
            geometry.Point(slope + 3.0 * slope_std, 1.0),
            geometry.Point(slope - 3.0 * slope_std, 1.0),
        ]
        max_abs_slope = max(max_abs_slope, abs(slope))

        polygon = geometry.Polygon([[point.x, point.y] for point in points_region])
        polygons_region.append(polygon)

        polygon = geometry.Polygon([[point.x, point.y] for point in points_slope])
        polygons_slope.append(polygon)

    intersection = polygons_region[0]
    union = polygons_region[0]
    for polygon in polygons_region[1::]:
        intersection = intersection.intersection(polygon)
        union = union.union(polygon)
    area_intersection = intersection.area / union.area

    intersection = polygons_slope[0]
    union = polygons_slope[0]
    for polygon in polygons_slope[1::]:
        intersection = intersection.intersection(polygon)
        union = union.union(polygon)
    slope_intersection = intersection.area / union.area

    metrics_dict['area_intersection'].append(area_intersection)
    metrics_dict['slope_intersection'].append(slope_intersection)
    metrics_dict['max_abs_slope'].append(max_abs_slope)


def process_variance_polygon(
    configs_child,
    item,
    xs,
    metrics_dict
):
    xs_all = []
    ys_b_all = []
    ys_t_all = []

    left_x = float('-inf')
    right_x = float('inf')

    for config_child_id, config_child in enumerate(configs_child):

        targets = xs[config_child_id]

        xs_curr = get_box_xs(targets)
        xs_all.append(xs_curr)

        item_id = config_child.advanced_dict[item]
        metrics_dict = dict_slice(config_child.advanced_data, item_id)

        ys_b, ys_t = fit_variance(xs_curr, metrics_dict)
        ys_b_all.append(ys_b)
        ys_t_all.append(ys_t)

        if (xs_curr[0] > left_x):
            left_x = xs_curr[0]
        if (xs_curr[-1] < right_x):
            right_x = xs_curr[-1]

    begin_ids = []
    end_ids = []
    for child_id in range(0, len(xs_all)):
        begin_ids.append(find_nearest_id(xs_all[child_id], left_x))
        end_ids.append(find_nearest_id(xs_all[child_id], right_x))

    polygons = []
    increasings = []

    for child_id in range(0, len(xs_all)):

        begin_id = begin_ids[child_id]
        end_id = end_ids[child_id]

        points = []
        for p_id in range(begin_id, end_id + 1):
            points.append(geometry.Point(
                xs_all[child_id][p_id],
                ys_t_all[child_id][p_id]
            ))
        for p_id in range(end_id, begin_id - 1, -1):
            points.append(geometry.Point(
                xs_all[child_id][p_id],
                ys_b_all[child_id][p_id]
            ))
        polygon = geometry.Polygon([[point.x, point.y] for point in points])
        polygons.append(polygon)

        diff_begin = abs(ys_t_all[child_id][begin_id] - ys_b_all[child_id][begin_id])
        diff_end = abs(ys_t_all[child_id][end_id] - ys_b_all[child_id][end_id])
        if diff_begin > np.finfo(float).eps and diff_end > np.finfo(float).eps:
            increasings.append(max(diff_end / diff_begin, diff_begin / diff_end))
        else:
            increasings.append(0.0)

    all_polygons_are_valid = True
    for polygon in polygons:
        if polygon.is_valid is False:
            all_polygons_are_valid = False
            break
    if all_polygons_are_valid:
        intersection = polygons[0]
        union = polygons[0]
        for polygon in polygons[1::]:
            intersection = intersection.intersection(polygon)
            union = union.union(polygon)
        area_intersection = intersection.area / union.area
        increasing = max(increasings) / min(increasings)
        increasing_id = np.argmax(increasings)
    else:
        area_intersection = 1.0
        increasing = 0.0
        increasing_id = 0

    metrics_dict['area_intersection'].append(area_intersection)
    metrics_dict['increasing'].append(increasing)
    metrics_dict['increasing_id'].append(increasing_id)
