import numpy as np
from sklearn.cluster import DBSCAN
from pydnameth.routines.common import normalize_to_0_1


def process_cluster(x, y, method_params, metrics_dict, suffix):

    x_normed = normalize_to_0_1(x)
    y_normed = normalize_to_0_1(y)

    min_samples = max(1, int(method_params['min_samples_percentage'] * len(x) / 100.0))

    X = np.array([x_normed, y_normed]).T
    db = DBSCAN(eps=method_params['eps'], min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    number_of_noise_points = list(labels).count(-1)
    percent_of_noise_points = float(number_of_noise_points) / float(len(x)) * 100.0

    metrics_dict['number_of_clusters' + suffix].append(number_of_clusters)
    metrics_dict['number_of_noise_points' + suffix].append(number_of_noise_points)
    metrics_dict['percent_of_noise_points' + suffix].append(percent_of_noise_points)
