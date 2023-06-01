import matplotlib.pyplot as plt
from itertools import groupby
from typing import Dict, List, Tuple


# -------------------------functions for plotting sequential results-------------------------


def plot_results_seq(results_n: Dict[str, List[Tuple]], results_h: Dict[str, List[Tuple]]) -> None:
    seq_results_normal = _filter_info_seq(_get_info_seq(results_n))
    seq_results_heuristic = _filter_info_seq(_get_info_seq(results_h))
    plot_seq(seq_results_normal, seq_results_heuristic)


def _get_info_seq(seq_results: Dict[str, List[Tuple]]) -> Dict[str, List]:
    """
    extract information dictionary from each tuple
    """
    new_seq_results = {}
    for ds_name, tuples_list in seq_results.items():
        new_seq_results[ds_name] = [t[2] for t in tuples_list]
    return new_seq_results


def _filter_info_seq(seq_results: Dict[str, List]) -> Dict[str, List[Dict]]:
    """
    remove unnecessary fields from each information dictionary
    """
    new_seq_results = {}
    for ds_name, info_list in seq_results.items():
        new_info_list = []
        for info in info_list:
            new_info = {
                'sample_name': info['sample_name'],
                'threshold': info['threshold'],
                'total_time': info['total_time']
            }
            new_info_list.append(new_info)
        new_seq_results[ds_name] = new_info_list
    return new_seq_results


def plot_seq(seq_results_normal: Dict[str, List[Dict]], seq_results_heuristic: Dict[str, List[Dict]]) -> None:
    """
    plot total time against threshold for all ds_name
    """
    fig, ax = plt.subplots()
    for ds_name, info_list in seq_results_normal.items():
        x = [info['threshold'] for info in info_list]
        y = [info['total_time'] for info in info_list]
        ax.plot(x, y, marker='o', label=ds_name)

    for ds_name, info_list in seq_results_heuristic.items():
        x = [info['threshold'] for info in info_list]
        y = [info['total_time'] for info in info_list]
        ax.plot(x, y, marker='o', label=ds_name + ' (heuristic)')

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Total Time')
    ax.set_title('Total Time vs. Threshold by Dataset')
    ax.legend()
    plt.show()


# -------------------------functions for plotting parallel with MapReduce and Spark results-------------------------


def plot_results_mr(results_n: Dict[str, Dict[str, List[Tuple]]], results_h: Dict[str, Dict[str, List[Tuple]]]) -> None:
    mr_results = _process_results_mr(results_n)
    mr_results_h = _process_results_mr(results_h)
    thresholds = _get_thresholds_mr(mr_results)
    thresholds_h = _get_thresholds_mr(mr_results_h)

    if thresholds != thresholds_h:
        raise ValueError("The two results don't use the same thresholds")
    plot_mr(mr_results, mr_results_h, thresholds)


def _process_results_mr(all_mr_results: Dict[str, Dict[str, List[Tuple]]]) -> Dict[str, Dict[str, List]]:
    return _group_by_n_execs_mr(_get_info_mr(all_mr_results))


def _get_info_mr(all_mr_results: Dict[str, Dict[str, List[Tuple]]]) -> Dict[str, Dict[str, List]]:
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


def _group_by_n_execs_mr(all_mr_results: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, List]]:
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


def _get_thresholds_mr(processed_dict: Dict[str, Dict[str, List]]) -> List:
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


def _extract_threshold_dict_mr(processed_dict: Dict[str, Dict[str, List]], threshold):
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


def plot_mr(normal_dict: Dict[str, Dict[str, List]], heuristic_dict: Dict[str, Dict[str, List]],
            threshold_list: List) -> None:
    """
    plot the total time against the number of executors for different datasets, for all thresholds
    :param normal_dict: dictionary containing the normal case data
    :param heuristic_dict: dictionary containing the heuristic case data
    :param threshold_list:
    :return:
    """
    n_plots = len(threshold_list)
    n_rows = (n_plots + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 6 * n_rows))
    axs = axs.flatten()

    for i, threshold in enumerate(threshold_list):
        ax = axs[i]
        threshold_dict_normal = _extract_threshold_dict_mr(normal_dict, threshold)
        threshold_dict_heuristic = _extract_threshold_dict_mr(heuristic_dict, threshold)

        ax.set_title(f"Threshold {threshold}")
        ax.set_xlabel("Number of Executors")
        ax.set_ylabel("Total Time [seconds]")

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
