import sys
import os
import numpy as np
import time
import pickle
import itertools
import csv
import findspark
from typing import Tuple, List, Dict
from tqdm import tqdm

from pyspark import SparkFiles
from pyspark.sql import SparkSession
from sklearn.metrics.pairwise import cosine_similarity

from loguru import logger
from scipy.sparse import csr_matrix, vstack

# set the python executable of the current conda environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
findspark.init()

MASTER_HOST = "localhost"


def exec_mr_apds(
        tfidf_docs: List[Tuple[str, csr_matrix, float, Dict]],
        thresholds: List[float],
        n_executors: List[int],
        n_slices: List[int],
        mr_results_path: str,
        samples_dir: str,
        heuristic: bool = False) -> Dict:
    """
    wrapper function to increase readability of the notebook
    execute the parallel algorithm with spark and MapReduce
    :param tfidf_docs:
    :param thresholds:
    :param n_executors:
    :param n_slices:
    :param mr_results_path:
    :param samples_dir:
    :param heuristic:
    :return:
    """
    all_mr_results = {}

    if not os.path.exists(mr_results_path):
        for name, vectorized_docs, tfidf_time, idx_to_id in tqdm(tfidf_docs):
            mr_results_th = {}
            for threshold in thresholds:
                mr_results = []
                for workers in n_executors:
                    for slices in n_slices:
                        print(f'\n----working on {name}, threshold: {threshold}----')
                        print(f'\n----workers: {workers}, slices: {slices}----')

                        doc_pairs, doc_pairs_id, doc_pairs_info = spark_apds(ds_name=name,
                                                                             tfidf_mat=vectorized_docs,
                                                                             idx_to_id=idx_to_id,
                                                                             threshold=threshold,
                                                                             tfidf_time=tfidf_time,
                                                                             n_executors=workers,
                                                                             n_slices=slices,
                                                                             heuristic=heuristic)
                        mr_results.append((doc_pairs, doc_pairs_id, doc_pairs_info))
                        save_mr_result_csv(all_pairs=doc_pairs_id,
                                           ds_name=name,
                                           threshold=threshold,
                                           executors=workers,
                                           samples_dir=samples_dir,
                                           heuristic=heuristic)
                        print('----Done----')
                    print('\n')
                mr_results_th[threshold] = mr_results
            all_mr_results[name] = mr_results_th

        with open(mr_results_path, "wb") as f:
            pickle.dump(all_mr_results, f)
    else:
        with open(mr_results_path, "rb") as f:
            all_mr_results = pickle.load(f)

    return all_mr_results


def spark_apds(
        ds_name: str,
        tfidf_mat: csr_matrix,
        idx_to_id: Dict,
        threshold: float,
        tfidf_time: float,
        n_executors: int,
        n_slices: int,
        heuristic: bool = False
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[str, str, float]], Dict[str, object]]:
    """
    Parallelized version of All Pairs Documents Similarity with PySpark and MapReduce
    :param ds_name: dataset name
    :param tfidf_mat: tfidf vectorized documents
    :param idx_to_id: original mapping
    :param threshold:
    :param tfidf_time:
    :param n_executors: number of executors
    :param n_slices: number of rdd's partitions
    :param heuristic:
    :return: tuple containing:
    document pairs and their similarity, document pairs mapped to the actual document ids,
    and a dictionary of dataset name, threshold, number of pairs, execution time, n_slices, n_executors
    """

    def _apds_map(doc_info: Tuple[int, np.ndarray]) -> List[Tuple[int, Tuple[int, np.ndarray]]]:
        """
        Mapping function
        :param doc_info: tuple containing the doc_id and the corresponding TF-IDF vector
        :return: list of key-value pairs:
                    - key is the term_id
                    - value is a tuple of doc_id and the corresponding TF-IDF vector
        """
        doc_id, tfidf_entries = doc_info
        result = []

        for term_id in np.nonzero(tfidf_entries)[0]:
            if term_id > b_d.value[doc_id]:
                result.append((term_id, (doc_id, tfidf_entries)))
        return result

    def _apds_reduce(term_info: Tuple[int, List[Tuple[int, np.ndarray]]]) -> List[Tuple[int, int, float]]:
        """
        Reduce function
        :param term_info: tuple of term_id, and list of doc_id and corresponding TF-IDF vector
        :return: list of tuples of doc_id1 and doc_id2, and their similarity
        """
        term, tfidf_entries = term_info
        result = []

        for (doc_id1, doc_1), (doc_id2, doc_2) in itertools.combinations(tfidf_entries, 2):
            if term == np.max(np.intersect1d(np.nonzero(doc_1), np.nonzero(doc_2))):
                sim = cosine_similarity([doc_1], [doc_2])[0][0]
                if sim >= sim_threshold.value:
                    result.append((doc_id1, doc_id2, sim))
        return result

    def _apds_reduce_h(term_info: Tuple[int, List[Tuple[int, np.ndarray]]]) -> List[Tuple[int, int, float]]:
        """
        Reduce function with heuristic that filters out document pairs that are unlikely to be similar
        :param term_info: tuple of term_id, and list of doc_id and corresponding TF-IDF vector
        :return: list of tuples of doc_id1 and doc_id2, and their similarity
        """
        term, tfidf_entries = term_info
        result = []

        for (doc_id1, doc_1), (doc_id2, doc_2) in itertools.combinations(tfidf_entries, 2):
            # Skip comparison if the heuristic is enabled and the length difference is too large
            if abs(len(doc_1) - len(doc_2)) >= max(len(doc_1), len(doc_2)) / 2:
                continue
            if term == np.max(np.intersect1d(np.nonzero(doc_1), np.nonzero(doc_2))):
                sim = cosine_similarity([doc_1], [doc_2])[0][0]
                if sim >= sim_threshold.value:
                    result.append((doc_id1, doc_id2, sim))
        return result

    def _compute_b_d(docs: list[tuple[int, csr_matrix]], d_star: csr_matrix, threshold: float) -> Dict[int, int]:
        """
        For prefix filter. Given a threshold and d_star, computes the document boundaries
        :param docs: A NumPy array of tuples, where each tuple contains a doc_id and the corresponding TF-IDF vector
        :param d_star: maximum document, contains the maximum score of term in any document
        :param threshold:
        :return: dictionary of boundaries, where the keys are doc_id
        and the values are the indices of the largest term in each document right before the threshold is reached.
        """
        b_d = {}

        for doc_id, tfidf_entry in docs:
            tmp_dot = 0
            for pos, val in enumerate(tfidf_entry.data):
                tmp_dot += val * d_star[0, tfidf_entry.indices[pos]]
                if tmp_dot >= threshold:
                    b_d[doc_id] = pos - 1
                    break
            if doc_id not in b_d.keys():
                b_d[doc_id] = len(tfidf_entry.data) - 1
        return b_d

    def _compute_d_star(sorted_mat: csr_matrix) -> csr_matrix:
        """
        For prefix filter, Gete scr_matrix where the i-th entry is the maximum value of the i-th column (term)
        :param sorted_mat:
        :return:
        """
        return vstack(sorted_mat.max(axis=0).toarray())

    # Create SparkSession
    spark = create_spark_session(app_name="mr_all_pairs_docs_similarity")

    # Get sparkContext
    sc = spark.sparkContext

    # sort terms in decreasing order by document frequency
    tfidf_array = tfidf_mat.toarray()
    doc_freq = np.sum(tfidf_array > 0, axis=0)
    sorted_doc_freq = np.argsort(doc_freq)[::-1]
    sorted_tfidf_mat = tfidf_mat[:, sorted_doc_freq]

    # create the RDD: a list of pairs of document id and the corresponding TF-IDF vector
    processed_mat = list(zip(range(sorted_tfidf_mat.shape[0]), sorted_tfidf_mat))
    rdd = sc.parallelize(processed_mat, numSlices=n_slices * n_executors).persist()

    # broadcast read-only variables
    sim_threshold = sc.broadcast(threshold)
    d_star = _compute_d_star(sorted_tfidf_mat)  # computing d*
    b_d = sc.broadcast(_compute_b_d(processed_mat, d_star, threshold))  # compute b_d

    # define operations on rdd
    out = (
        rdd
        # perform mapping
        .flatMap(_apds_map)
        # group by term_id
        .groupByKey()
        # perform reduce
        .flatMap(_apds_reduce_h if heuristic else _apds_reduce)
        # remove duplicates
        .distinct()
    )

    start = time.time()
    reduced_results = out.collect()
    exec_time = time.time() - start

    # Stop spark session
    spark.stop()

    return (
        reduced_results,
        [(idx_to_id[id_1], idx_to_id[id_2], sim) for (id_1, id_2, sim) in reduced_results],
        {
            'sample_name': ds_name,
            'threshold': threshold,
            'pairs_count': len(reduced_results),
            'tfidf_time': tfidf_time,
            'exec_time': exec_time,
            'total_time': exec_time + tfidf_time,
            'n_slices': n_slices,
            'n_executors': n_executors
        }
    )


def create_spark_session(app_name: str) -> SparkSession:
    spark: SparkSession = (
        SparkSession.builder
        .master(f"spark://{MASTER_HOST}:7077")
        .appName(f"{app_name}")
        # .config("spark.driver.memory", "10g")
        # .config("spark.executor.cores", "1")
        # .config("spark.executor.memory", "10g")
        .getOrCreate()
    )

    # Add local dependencies (local python source files) to SparkContext and sys.path
    src_zip_path = os.path.abspath("../../src.zip")
    logger.debug(f"Adding {src_zip_path} to SparkContext")
    spark.sparkContext.addPyFile(src_zip_path)
    sys.path.insert(0, SparkFiles.getRootDirectory())

    return spark


def save_mr_result_csv(all_pairs: List[Tuple[str, str, float]],
                       ds_name: str,
                       threshold: float,
                       executors: int,
                       samples_dir: str,
                       heuristic: bool = False) -> None:
    """
    save the document pairs and their similarity sorted by their similarity as a .csv file
    :param all_pairs: list of unique similar pair with the similarity
    :param ds_name: dataset name
    :param threshold: threshold used
    :param executors: number of executors used
    :param samples_dir: directory path of the ds_name sample
    :param heuristic: if the heuristic was used in computing the pairs
    :return:
    """
    save_dir = os.path.join(samples_dir, "mr_result", ds_name, f'{threshold}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = f'{executors}_workers_h.csv' if heuristic else f'{executors}_workers.csv'
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(all_pairs)
