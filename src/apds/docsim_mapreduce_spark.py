import sys
import os
from typing import Tuple, Type, List, Dict, Any
import numpy as np
import time
import itertools
import csv
import findspark

from pyspark import SparkFiles
from pyspark.sql import SparkSession
from sklearn.metrics.pairwise import cosine_similarity

from loguru import logger
from scipy.sparse import csr_matrix

# Needed to correctly set the python executable of the current conda environment
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
findspark.init()

MASTER_HOST = "localhost"


def spark_apds(ds_name: str,
               sampled_dict: csr_matrix,
               threshold: float,
               n_executors: int,
               n_slices: int,
               heuristic: bool):  # -> Tuple[List[Tuple[str, str, float]], Tuple[str, float, float, int, int]]:
    """
    PURPOSE: perform PySpark version of All Pairs Documents Similarity
    ARGUMENTS:
        - ds_name (str): Dataset name
        - sampled_dict (Dict[str, str]): sampled documents
        - threshold (float): threshold to use
        - workers (int): number of workers to use
        - s_factor (int): numSlice factor
    RETURN:
        - (Tuple[List[Tuple[str, str, float]], Tuple[str, loat, float, int, int]])
            - List of tuples of similar unique pair with the relative similarity
            - [ds_name, elapsed, threshold, uniqie_pairs_sim_docs, workers]
    """

    # Map
    def _apds_map(pair: Tuple[int, np.ndarray]) -> List[Tuple[int, Tuple[int, np.ndarray]]]:
        """
        PURPOSE: apply map to the RDD
        ARGUMENTS:
            - pair (Tuple[int, np.ndarray]):
                   tuple of docid and TF-IDF np.ndarray for the relative document
        RETURN:
            - (List[Tuple[int, Tuple[int, np.ndarray]]]):
                   list of pairs of termid and docid and TF-IDF np.ndarray pair
        """

        docid, tf_idf_list = pair
        res = []
        for id_term in np.nonzero(tf_idf_list)[0]:
            if id_term > sc_b_d.value[docid]:
                res.append((id_term, (docid, tf_idf_list)))
        return res

    # Reduce
    def _apds_reduce(pair: Tuple[int, List[Tuple[int, np.ndarray]]]) -> List[Tuple[int, int, float]]:
        """
        PURPOSE: apply reduce to the RDD
        ARGUMENTS:
            - pair (Tuple[int, List[Tuple[int, np.ndarray]]]):
                tuple of termid, list of pairs of docid and TF-IDF np.ndarray
        RETURN:
            - (List[Tuple[int, int, float]]):
                   list of tuples of termid and docid_1, docid_2 and similarity
        """
        term, tf_idf_list = pair
        res = []
        # Use itertools.combinations to perform smart nested for loop
        for (id1, d1), (id2, d2) in itertools.combinations(tf_idf_list, 2):
            # HEURISTIC - skip if too-high length mismatch
            # if len(d1) / len(d2) > 1.5:
            #     break
            if term == np.max(np.intersect1d(np.nonzero(d1), np.nonzero(d2))):
                sim = cosine_similarity([d1], [d2])[0][0]
                if sim >= sc_treshold.value:
                    res.append((id1, id2, sim))
        return res

    def _compute_b_d(docs: np.ndarray, d_star: np.ndarray, threshold: float) -> Dict[int, int]:
        """
        PURPOSE:
        ARGUMENTS:
            - docs (np.ndarray): TF-IDF matrix
            - d_star (np.ndarray)
            - threshold (flaot)
        RETURN:
            - (Dict[int, int]) prefix filter result
        """

        b_d = {}
        for doc_id, tfidf_row in docs:
            temp_product_sum = 0
            for pos, tfidf_val in enumerate(tfidf_row):
                temp_product_sum += tfidf_val * d_star[pos]
                if temp_product_sum >= threshold:
                    b_d[doc_id] = pos - 1
                    break
            if doc_id not in list(b_d.keys()):
                b_d[doc_id] = len(tfidf_row) - 1
        return b_d

    # Create SparkSession
    spark = create_spark_session(app_name="mr_all_pairs_docs_similarity")

    # Get sparkContext
    sc = spark.sparkContext

    tfidf_features = sampled_dict.toarray()  # Get the TF-IDF matrix.toarray

    doc_freq = np.sum(tfidf_features > 0, axis=0)  # Compute document frequency
    dec_doc_freq = np.argsort(doc_freq)[::-1]  # Decreasing order of document frequency
    # Order the matrix with the index of the decreasing order of document frequency
    matrix = np.array([row[dec_doc_freq] for row in tfidf_features])

    # 1) Zip each document id with its vectorial representation
    # Computing the list that will feed into the rdd, list of pairs of (docid, tfidf_list)
    list_pre_rrd = list(zip(range(len(tfidf_features)), matrix))
    rdd = sc.parallelize(list_pre_rrd, numSlices=n_slices * n_executors).persist()  # Create the RDD
    # same as this
    # vectorized_docs_rdd = sc.parallelize(zip(keys, doc_matrix), n_workers * n_slices).persist()

    # 2) broadcast variables to be used by spark
    # Broadcasting more efficient than passing it
    sc_treshold = sc.broadcast(threshold)

    d_star = np.max(matrix.T, axis=1)  # Computing d*
    sc_b_d = sc.broadcast(_compute_b_d(list_pre_rrd, d_star, threshold))  # Compute and propagate the b_d

    # 3) Compute with spark:
    #   1. MAP function using flatMap(MAP).
    #   2. Group by term id.
    #   3. REDUCE function using:
    #       1. flatMap(filter_pairs)
    #       2. map(compute_similarity)
    #       3. filter(similar_doc)
    # Adding all transformations

    out = (
        rdd
        # perform mapping
        .flatMap(_apds_map)
        # combine by key
        .groupByKey()
        # perform reduce
        .flatMap(_apds_reduce)
        # remove duplicates
        .distinct()
    )

    start = time.time()
    reduced_results = out.collect()  # Collection the result
    end = time.time()

    spark.stop()  # Stop spark session

    # doc_keys = list(sampled_dict.keys())  # used in the end to get the ids i think

    # print(f"Dataset: {ds_name} - Threshold: {threshold} - worker and pslit i guess: {n_workers} {n_slices}")
    # print(f"Exec time: {end - start}")
    # print(f"Docs {reduced_results}")

    return reduced_results, (end - start)
    # return [(doc_keys[id1], doc_keys[id2], sim) for (id1, id2, sim) in reduced_results], \
    #        [ds_name, end - start, threshold, len(reduced_results), n_workers, n_slices]


def create_spark_session(app_name: str) -> SparkSession:
    spark: SparkSession = (
        SparkSession.builder
        .master(f"spark://{MASTER_HOST}:7077")
        .appName(f"{app_name}")
        # .config("spark.driver.memory", "1g")
        # .config("spark.executor.cores", "1")
        # .config("spark.executor.memory", "1g")
        .getOrCreate()
    )

    # Add local dependencies (local python source files) to SparkContext and sys.path
    src_zip_path = os.path.abspath("../../src.zip")
    logger.debug(f"Adding {src_zip_path} to SparkContext")
    spark.sparkContext.addPyFile(src_zip_path)
    sys.path.insert(0, SparkFiles.getRootDirectory())

    return spark


def create_doc_sim_csv(pairs_list: List[Tuple[str, str, float]], ds_name: str,
                       threshold: float, workers: int, samples_dir) -> None:
    """
    PURPOSE: create the .csv file sotring the list of similar documents pairs with the cosine similarity
    ARGUMENTS:
        - pairs_list (List[Tuple[str, str, float]]): list of unique similar pair with the similarity
        - ds_name: (str): dataset name
        - threshold (float): used threshold
        - workers (None | int): number of workers used
    RETURN: None
    """

    save_dir = os.path.join(samples_dir, "mr_result", ds_name, f'{threshold}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, f'{workers}_workers.csv')
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(pairs_list)
