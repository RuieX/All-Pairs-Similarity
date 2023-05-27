from typing import Dict, List, Tuple
import findspark
findspark.init()

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
import itertools


def compute_b_d(matrix: np.ndarray, d_star: np.ndarray, threshold: float) -> Dict[int, int]:
    '''
    PURPOSE: download the dataset
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
        if(docid not in list(b_d.keys())):
            b_d[docid] = len(tfidf_row)-1
    return b_d



def pyspark_APDS(ds_name: str, sampled_dict: Dict[str, str], threshold: float,
                 workers: int, s_factor: int) -> Tuple[List[Tuple[str, str, float]], Tuple[str, float, float, int, int]]:
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
            if term == np.max(np.intersect1d(np.nonzero(d1), np.nonzero(d2))):
                sim = cosine_similarity([d1], [d2])[0][0]
                if sim >= sc_treshold.value:
                    res.append((id1, id2, sim))
        return res


    # Create SparkSession
    spark = SparkSession \
        .builder \
        .config(conf = SparkConf().setMaster(f"local[{workers}]") \
                .setAppName("all_pairs_docs_similarity.com") \
                .set("spark.executor.memory", "10g") \
                .set("spark.executor.cores", "1") \
                .set("spark.driver.memory", "10g")) \
        .getOrCreate()

    sc = spark.sparkContext # Get sparkContext

    sc_treshold = sc.broadcast(threshold) # Broadcasting the threshold

    doc_keys = list(sampled_dict.keys())

    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(sampled_dict).toarray() # Get the TF-IDF matrix

    doc_freq = np.sum(tfidf_features > 0, axis=0) # Compute document frequency
    dec_doc_freq = np.argsort(doc_freq)[::-1] # Decreasing order of document frequency

    # Order the matrix with the index of the decreasing order of document frequency
    matrix = np.array([row[dec_doc_freq] for row in tfidf_features])

    # Computing the list that will feed into the rdd, list of pairs of (docid, tfidf_list)
    list_pre_rrd = list(zip(range(len(tfidf_features)), matrix))

    d_star = np.max(matrix.T, axis=1) # Computing d*
    sc_b_d = sc.broadcast(compute_b_d(list_pre_rrd, d_star, threshold)) # Compute and propagate the b_d

    rdd = sc.parallelize(list_pre_rrd, numSlices=s_factor*workers) # Creare the RDD

    # Adding all transformations
    reduced = rdd.flatMap(map_fun).groupByKey().flatMap(reduce_fun).persist()

    start = time.time()
    reduced_results = reduced.collect() # Collection the result
    end = time.time()

    spark.stop() # Stop spark session

    return [(doc_keys[id1], doc_keys[id2], sim) for (id1, id2, sim) in reduced_results], \
           [ds_name, end-start, threshold, len(reduced_results), workers, s_factor]
