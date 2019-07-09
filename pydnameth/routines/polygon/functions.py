import numpy as np
from shapely import geometry


def process_variance_polygon(
    begin_ids,
    end_ids,
    x_range,
    xs_all,
    ys_b_real_all,
    ys_t_real_all,
    ys_b_fit_all,
    ys_t_fit_all,
    suffix,
    metrics_dict
):
    polygons_fit = []
    polygons_real = []
    increasings_fit = []
    increasings_real = []

    for child_id in range(0, len(xs_all)):

        begin_id = begin_ids[child_id]
        end_id = end_ids[child_id]

        points_fit = []
        points_real = []
        for p_id in range(begin_id, end_id + 1):
            points_fit.append(geometry.Point(
                xs_all[child_id][p_id],
                ys_t_fit_all[child_id][p_id]
            ))
            points_real.append(geometry.Point(
                xs_all[child_id][p_id],
                ys_t_real_all[child_id][p_id]
            ))
        for p_id in range(end_id, begin_id - 1, -1):
            points_fit.append(geometry.Point(
                xs_all[child_id][p_id],
                ys_b_fit_all[child_id][p_id]
            ))
            points_real.append(geometry.Point(
                xs_all[child_id][p_id],
                ys_b_real_all[child_id][p_id]
            ))
        polygon = geometry.Polygon([[point.x, point.y] for point in points_fit])
        polygons_fit.append(polygon)
        polygon = geometry.Polygon([[point.x, point.y] for point in points_real])
        polygons_real.append(polygon)

        diff_begin_fit = abs(ys_t_fit_all[child_id][begin_id] - ys_b_fit_all[child_id][begin_id])
        diff_end_fit = abs(ys_t_fit_all[child_id][end_id] - ys_b_fit_all[child_id][end_id])
        if diff_begin_fit > np.finfo(float).eps and diff_end_fit > np.finfo(float).eps:
            increasings_fit.append(max(diff_end_fit / diff_begin_fit, diff_begin_fit / diff_end_fit))
        else:
            increasings_fit.append(0.0)

        diff_begin_real = abs(ys_t_real_all[child_id][begin_id] - ys_b_real_all[child_id][begin_id])
        diff_end_real = abs(ys_t_real_all[child_id][end_id] - ys_b_real_all[child_id][end_id])
        if diff_begin_real > np.finfo(float).eps and diff_end_real > np.finfo(float).eps:
            increasings_real.append(max(diff_end_real / diff_begin_real, diff_begin_real / diff_end_real))
        else:
            increasings_real.append(0.0)

    all_polygons_fit_is_valid = True
    for polygon in polygons_fit:
        if polygon.is_valid is False:
            all_polygons_fit_is_valid = False
            break
    if all_polygons_fit_is_valid:
        intersection = polygons_fit[0]
        union = polygons_fit[0]
        for polygon in polygons_fit[1::]:
            intersection = intersection.intersection(polygon)
            union = union.union(polygon)
        area_intersection_fit = intersection.area / union.area
        increasing_fit = max(increasings_fit) / min(increasings_fit)
        increasing_fit_id = np.argmax(increasings_fit)
    else:
        area_intersection_fit = 1.0
        increasing_fit = 0.0
        increasing_fit_id = 0
    metrics_dict['area_intersection_fit' + suffix].append(area_intersection_fit)
    metrics_dict['increasing_fit' + suffix].append(increasing_fit)
    metrics_dict['increasing_fit_normed' + suffix].append(increasing_fit / x_range)
    metrics_dict['increasing_fit_id' + suffix].append(increasing_fit_id)

    all_polygons_real_is_valid = True
    for polygon in polygons_real:
        if polygon.is_valid is False:
            all_polygons_real_is_valid = False
            break
    if all_polygons_real_is_valid:
        intersection = polygons_real[0]
        union = polygons_real[0]
        for polygon in polygons_real[1::]:
            intersection = intersection.intersection(polygon)
            union = union.union(polygon)
        area_intersection_real = intersection.area / union.area
        increasing_real = max(increasings_real) / min(increasings_real)
        increasing_real_id = np.argmax(increasings_real)
    else:
        area_intersection_real = 1.0
        increasing_real = 0.0
        increasing_real_id = 0
    metrics_dict['area_intersection_real' + suffix].append(area_intersection_real)
    metrics_dict['increasing_real' + suffix].append(increasing_real)
    metrics_dict['increasing_real_normed' + suffix].append(increasing_real / x_range)
    metrics_dict['increasing_real_id' + suffix].append(increasing_real_id)
