import sys
import os
from typing import Tuple, Type, List, Dict, Any
import numpy as np
import time
import itertools
import csv
import findspark
findspark.init()

import pyspark.sql as psql
from pyspark import SparkFiles
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from loguru import logger
from scipy.sparse import csr_matrix

# IMPORTANT: create session prior to importing pyspark.pandas, else
#   spark won't use all specified cores
from src.utilities.utils import AVAILABLE_CORES, AVAILABLE_RAM_GB

# Needed to correctly set the python executable of the current conda env
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ['SPARK_LOCAL_IP'] = "127.0.0.1"

# UserWarning: 'PYARROW_IGNORE_TIMEZONE' environment variable was not set.
# It is required to set this environment variable to '1' in both driver and executor
#   sides if you use pyarrow>=2.0.0.
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

MASTER_HOST = "localhost"  # master host in local standalone cluster


def compute_b_d(matrix: np.ndarray, d_star: np.ndarray, threshold: float) -> Dict[int, int]:
    '''
    PURPOSE:
    ARGUMENTS:
        - matrix (np.ndarray): TF-IDF matrix
        - d_star (np.ndarray)
        - threshold (flaot)
    RETURN:
        - (Dict[int, int]) prefix filter result
    '''

    b_d = {}
    for docid, tfidf_row in matrix:
        temp_product_sum = 0
        for pos, tfidf_val in enumerate(tfidf_row):
            temp_product_sum += tfidf_val * d_star[pos]
            if temp_product_sum >= threshold:
                b_d[docid] = pos - 1
                break
        if (docid not in list(b_d.keys())):
            b_d[docid] = len(tfidf_row) - 1
    return b_d


def pyspark_APDS(ds_name: str, sampled_dict: csr_matrix, threshold: float,
                 n_workers: int,
                 n_slices: int):  # -> Tuple[List[Tuple[str, str, float]], Tuple[str, float, float, int, int]]:
    '''
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
    '''

    # Map functuion
    def map_fun(pair: Tuple[int, np.ndarray]) -> List[Tuple[int, Tuple[int, np.ndarray]]]:
        '''
        PURPOSE: apply map to the RDD
        ARGUMENTS:
            - pair (Tuple[int, np.ndarray]):
                   tuple of docid and TF-IDF np.ndarray for the relative document
        RETURN:
            - (List[Tuple[int, Tuple[int, np.ndarray]]]):
                   list of pairs of termid and docid and TF-IDF np.ndarray pair
        '''

        docid, tf_idf_list = pair
        res = []
        for id_term in np.nonzero(tf_idf_list)[0]:
            if id_term > sc_b_d.value[docid]:
                res.append((id_term, (docid, tf_idf_list)))
        return res

    # Reduce function
    def reduce_fun(pair: Tuple[int, List[Tuple[int, np.ndarray]]]) -> List[Tuple[int, int, float]]:
        '''
        PURPOSE: apply reduce to the RDD
        ARGUMENTS:
            - pair (Tuple[int, List[Tuple[int, np.ndarray]]]):
                tuple of termid, list of pairs of docid and TF-IDF np.ndarray
        RETURN:
            - (List[Tuple[int, int, float]]):
                   list of tuples of termid and docid_1, docid_2 and similarity
        '''

        term, tf_idf_list = pair
        res = []
        # Use itertools.combinations to perform smart nested for loop
        for (id1, d1), (id2, d2) in itertools.combinations(tf_idf_list, 2):
            # HEURISTIC - skip if too-high length mismatch
            if len(d1) / len(d2) > 1.5:
                break
            if term == np.max(np.intersect1d(np.nonzero(d1), np.nonzero(d2))):
                sim = cosine_similarity([d1], [d2])[0][0]
                if sim >= sc_treshold.value:
                    res.append((id1, id2, sim))
        return res

    # Create SparkSession
    # spark = create_spark_session(n_executors=n_workers, app_name="MR-APDSS")  # TODO VEDI PIER
    spark = (SparkSession
             .builder
             .config(conf=SparkConf()
                     .setMaster(f"spark://{MASTER_HOST}:7077")
                     .setAppName("all_pairs_docs_similarity.com")
                     .set("spark.driver.memory", "10g")
                     .set("spark.executor.cores", "1")
                     .set("spark.executor.memory", "10g"))
             .getOrCreate()
             )

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
    rdd = sc.parallelize(list_pre_rrd, numSlices=n_slices * n_workers).persist()  # Create the RDD
    # same as this
    # vectorized_docs_rdd = sc.parallelize(zip(keys, doc_matrix), n_workers * n_slices).persist()

    # 2) broadcast variables to be used by spark
    # Broadcasting more efficient than passing it
    sc_treshold = sc.broadcast(threshold)

    d_star = np.max(matrix.T, axis=1)  # Computing d*
    sc_b_d = sc.broadcast(compute_b_d(list_pre_rrd, d_star, threshold))  # Compute and propagate the b_d

    # 3) Compute with spark:
    #   1. MAP function using flatMap(MAP).
    #   2. Group by term id.
    #   3. REDUCE function using:
    #       1. flatMap(filter_pairs)
    #       2. map(compute_similarity)
    #       3. filter(similar_doc)
    # Adding all transformations
    # out = (
    #     rdd
    #     .flatMap(map_fun)
    #     .groupByKey()
    #     .flatMap(reduce_fun)
    #     .persist()
    # )

    out = (
        rdd.
        # perform mapping
        flatMap(map_fun)
        # combine by key
        .groupByKey()
        # perform reduce
        .flatMapValues(reduce_fun)
        # remove duplicates
        .distinct()
    )

    # lotto
    # similarity_doc_pairs = vectorized_docs_rdd \
    #     .flatMap(MAP) \
    #     .groupByKey() \
    #     .flatMap(filter_pairs) \
    #     .map(compute_similarity) \
    #     .filter(similar_doc)

    start = time.time()
    reduced_results = out.collect()  # Collection the result
    end = time.time()

    spark.stop()  # Stop spark session

    doc_keys = list(sampled_dict.keys())  # used in the end to get the ids i think

    print(f"Dataset: {ds_name} - Threshold: {threshold} - worker and pslit i guess: {n_workers} {n_slices}")
    print(f"Exec time: {end - start}")
    print(f"Docs {reduced_results}")

    return reduced_results, (end - start)
    # return [(doc_keys[id1], doc_keys[id2], sim) for (id1, id2, sim) in reduced_results], \
    #        [ds_name, end - start, threshold, len(reduced_results), n_workers, n_slices]


def create_spark_session(n_executors: int, app_name: str) -> psql.SparkSession:
    driver_ram_gb = 2
    driver_cores = 2
    mem_per_executor = (AVAILABLE_RAM_GB - driver_ram_gb) // n_executors
    cores_per_executor = (AVAILABLE_CORES - driver_cores) // n_executors

    logger.debug(f"Executor memory: {mem_per_executor}")
    logger.debug(f"AVAILABLE_RAM_GB: {AVAILABLE_RAM_GB}")
    logger.debug(f"Total executor memory: {(AVAILABLE_RAM_GB - driver_ram_gb)}")
    logger.debug(f"Executor cores: {cores_per_executor}")

    spark = (SparkSession
             .builder
             .master(f"spark://{MASTER_HOST}:7077")
             .appName(f"{app_name}")
             .config(conf=SparkConf()
                     .set("spark.driver.memory", "10g")
                     .set("spark.executor.cores", "1")
                     .set("spark.executor.memory", "10g"))
             .getOrCreate()
             )

    # spark: psql.SparkSession = (
    #     psql.SparkSession.builder
    #     .master(f"spark://{MASTER_HOST}:7077")  # connect to previously started master host
    #
    #     .appName(f"{app_name}")
    #     # .config("spark.driver.host", f"{MASTER_HOST}:7077")
    #     .config("spark.driver.cores", driver_cores)
    #     .config("spark.driver.memory", f"{driver_ram_gb}g")
    #     .config("spark.executor.instances", n_executors)
    #     .config("spark.executor.cores", cores_per_executor)
    #     .config("spark.executor.memory", f"{mem_per_executor}g")
    #     .config("spark.default.parallelism", AVAILABLE_CORES)
    #     .config("spark.cores.max", AVAILABLE_CORES - driver_cores)
    #     .getOrCreate()
    # )

    # Add local dependencies (local python source files) to SparkContext and sys.path
    src_zip_path = os.path.abspath("../../src.zip")
    logger.debug(f"Adding {src_zip_path} to SparkContext")

    spark.sparkContext.addPyFile(src_zip_path)
    sys.path.insert(0, SparkFiles.getRootDirectory())

    return spark


def create_doc_sim_csv(pairs_list: List[Tuple[str, str, float]], ds_name: str,
                       threshold: float, workers: int, samples_dir) -> None:
    '''
    PURPOSE: create the .csv file sotring the list of similar documents pairs with the cosine similarity
    ARGUMENTS:
        - pairs_list (List[Tuple[str, str, float]]): list of unique similar pair with the similarity
        - ds_name: (str): dataset name
        - threshold (float): used threshold
        - workers (None | int): number of workers used
    RETURN: None
    '''

    save_dir = os.path.join(samples_dir, "mr_result", ds_name, f'{threshold}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(save_dir, f'{workers}_workers.csv')
    if not os.path.exists(path):
        with open(path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(pairs_list)
