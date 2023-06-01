import matplotlib.pyplot as plt
from itertools import groupby


def plot_results(results, results_h):
    mr_results = _process_results(results)
    mr_results_h = _process_results(results_h)
    thresholds = _get_thresholds(mr_results)
    thresholds_h = _get_thresholds(mr_results_h)

    if thresholds != thresholds_h:
        raise ValueError("The two results don't use the same thresholds")
    plot_threshold_dicts(mr_results, mr_results_h, thresholds)


def _process_results(all_mr_results):
    return _group_by_n_execs(_get_info_results(all_mr_results))


def _get_info_results(all_mr_results):
    """
    function to filter out all information not needed for plotting
    :param all_mr_results:
    :return:
    """
    all_info_results = {
        dataset_name: {
            threshold: [res[2] for res in mr_results]
            for threshold, mr_results in mr_results_th.items()
        }
        for dataset_name, mr_results_th in all_mr_results.items()
    }
    return all_info_results


def _group_by_n_execs(all_mr_results):
    """
    function to group the results by n_executors and get the average execution time
    :param all_mr_results:
    :return:
    """
    new_all_mr_results = {}
    for ds_name, mr_results_th in all_mr_results.items():
        new_mr_results_th = {}
        for threshold, mr_results in mr_results_th.items():
            new_mr_results = []
            for n_executors, result_dicts in groupby(
                    sorted(mr_results, key=lambda x: x['n_executors']),
                    key=lambda x: x['n_executors']
            ):
                total_time_sum = 0
                total_time_count = 0
                for result_dict in result_dicts:
                    total_time_sum += result_dict['total_time']
                    total_time_count += 1
                total_time_avg = total_time_sum / total_time_count
                new_result_dict = {
                    'sample_name': ds_name,
                    'threshold': threshold,
                    'total_time': total_time_avg,
                    'n_executors': n_executors
                }
                new_mr_results.append(new_result_dict)
            new_mr_results_th[threshold] = new_mr_results
        new_all_mr_results[ds_name] = new_mr_results_th
    return new_all_mr_results


def _get_thresholds(processed_dict):
    """
    function to extract the list of unique and sorted thresholds
    :param processed_dict:
    :return:
    """
    threshold_set = set()
    for ds_name in processed_dict.keys():
        for threshold in processed_dict[ds_name]:
            threshold_set.add(threshold)
    return sorted(list(threshold_set))


def _extract_threshold_dict(processed_dict, threshold):
    """
    function that given the threshold, extract from each ds_name the value (another dictionary) keyed by given threshold
    :param processed_dict:
    :param threshold:
    :return:
    """
    threshold_dict = {}
    for ds_name in processed_dict.keys():
        if threshold in processed_dict[ds_name]:
            threshold_dict[ds_name] = processed_dict[ds_name][threshold]
    return threshold_dict


def plot_threshold_dicts(normal_dict, heuristic_dict, threshold_list):
    """
    plot
    :param normal_dict:
    :param heuristic_dict:
    :param threshold_list:
    :return:
    """
    n_plots = len(threshold_list)
    n_rows = (n_plots + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows))
    axs = axs.flatten()

    for i, threshold in enumerate(threshold_list):
        ax = axs[i]
        threshold_dict_normal = _extract_threshold_dict(normal_dict, threshold)
        threshold_dict_heuristic = _extract_threshold_dict(heuristic_dict, threshold)

        ax.set_title(f"Threshold {threshold}")
        ax.set_xlabel("Number of Executors")
        ax.set_ylabel("Total Time")

        for ds_name, ds_data in threshold_dict_normal.items():
            x = []
            y = []
            for data_point in ds_data:
                if data_point["threshold"] == threshold:
                    x.append(data_point["n_executors"])
                    y.append(data_point["total_time"])
            ax.plot(x, y, marker='o', label=ds_name)

        for ds_name, ds_data in threshold_dict_heuristic.items():
            x = []
            y = []
            ds_name_h = ds_name + "_h"
            for data_point in ds_data:
                if data_point["threshold"] == threshold:
                    x.append(data_point["n_executors"])
                    y.append(data_point["total_time"])
            ax.plot(x, y, marker='o', label=ds_name_h)
        ax.legend()

    plt.tight_layout()
    plt.show()
