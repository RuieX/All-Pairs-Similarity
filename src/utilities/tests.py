# from typing import Dict, List, Tuple
# import findspark
# findspark.init()
#
# from pyspark.conf import SparkConf
# from pyspark.sql import SparkSession
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import time
# import itertools
#
#
# def document_map(
#         doc_info: tuple[str, int, list[tuple[int, float]]]
# ) -> list[tuple[str, tuple[str, int, list[tuple[int, float]]]]]:
#     """
#     Mapping function
#     :param doc_info: document information is represented as a triple:
#         - doc-id, represented as a string
#         - term-threshold, referring to the index of a specific column up to which do not map terms
#         - document vector, a sparse vector as a list of pairs (column, value) for each non-zero entries,
#             where the column is actually a term-id
#     :return: list of key-value pairs:
#         - key: term-id, which is actually a column index
#         - value: consists of a triple:
#             - doc-id  (the same as input)
#             - term-id (the same as the key)
#             - document vector (the same as input)
#     """
#
#     # unpacking
#     doc_id: str
#     term_threshold: int
#     sparse_entries: list[tuple[int, float]]
#     doc_id, term_threshold, sparse_entries = doc_info
#
#     mapped: list[tuple[str, tuple[str, int, list[tuple[int, float]]]]] = [
#
#         (str(term_id), (doc_id, term_id, sparse_entries))
#         for term_id, value in sparse_entries  # document terms by using non-zero entries
#         if term_id > term_threshold  # OPTIMIZATION 1:
#         # we only map term with higher term-id with respect to the threshold one
#         #  (thus, we only consider columns after the threshold one)
#     ]
#
#     return mapped
#
# def documents_reduce(docs: list[tuple[int, int, list[tuple[int, float]]]]) -> list[tuple[tuple[int, int], float]]:
#     """
#     Reduce function
#     :param docs: list of triplets:
#         - doc-id
#         - term-id (actually a column index of the vector)
#         - document vector as a sparse matrix of pairs (column, value)
#     :return: list of tuples:
#         - the first element is the pair of documents represented by their doc-id
#         - the second element represent their cosine-similarity
#     """
#
#     # list of output pairs
#     pairs = []
#
#     # DOC-T HEURISTIC pt. 1 - sort items for document length
#     docs = sorted(docs, key=lambda x: len(x[2]), reverse=True)
#
#     # total number of documents
#     n_docs = len(docs)
#
#     # loop among all possible pairs
#     for i in range(n_docs - 1):
#
#         doc1_id, term_id, doc1 = docs[i]
#
#         for j in range(i + 1, n_docs):
#
#             doc2_id, _, doc2 = docs[j]  # since the operation is an aggregation by key,
#             # term_id is expected to be the same
#
#             # DOC-TERMS HEURISTIC pt. 2 - skip if too-high length mismatch
#             if len(doc1) / len(doc2) > 1.3:
#                 break
#
#             # ----------------- OPTIMIZATION 2 -----------------
#
#             # collect term-ids of each document
#             terms_1: list[int] = [t_id1 for t_id1, _ in doc1]  # term-ids for the first document
#             terms_2: list[int] = [t_id2 for t_id2, _ in doc2]  # term-ids for the second document
#
#             # perform their intersection
#             common_terms: set[int] = set(terms_1).intersection(terms_2)
#
#             # get the maximum term-id
#             max_term: int = max(common_terms)
#
#             # if the maximum term-id is not the same of aggregation key, skip similarity computation
#             if term_id != max_term:
#                 pass
#
#             # --------------------------------------------------
#
#             # Computing similarity with dot-product
#
#             # getting iterator
#             iter_doc1 = iter(doc1)
#             iter_doc2 = iter(doc2)
#
#             # we assume documents with at least on term
#             term1, value1 = next(iter_doc1)
#             term2, value2 = next(iter_doc2)
#
#             sim = 0.  # total similarity
#
#             # we use iterators to keep a pointer over term-ids of the two vectors
#             # if they have the same term-id, we add its contribution to the cumulative sum and we move both pointers over
#             # otherwise we move over the one with smallest term-id
#
#             while True:
#
#                 try:
#                     if term1 == term2:  # they have common term-id; we add its contribution to final similarity
#                         sim += value1 * value2
#                         term1, value1 = next(iter_doc1)
#                         term2, value2 = next(iter_doc2)
#                     elif term1 < term2:  # the first one has a smaller term-id
#                         term1, value1 = next(iter_doc1)
#                     else:  # the second one has a smaller term-id
#                         term2, value2 = next(iter_doc2)
#                 except StopIteration:  # we scanned all terms of one of the vectors so there's no more term in common
#                     break
#
#             # we add the pairwise similarity to final output
#             pairs.append(((doc1_id, doc2_id), sim))
#
#     return pairs
#
#
# def MAINEQUESTODIOCANE():
#     """
#     Usage: pairwise_similarity [data_name] [similarity] [idf_order]
#     """
#
#     # creating spark session
#     spark = SparkSession\
#         .builder\
#         .appName(APP_NAME)\
#         .getOrCreate()
#
#     #
#     data_name = 'small'
#     similarity = 0.8
#     idf_order = True
#
#     # Getting input data
#     # docs_vet = DocumentVectors(data_name=data_name, idf_order=idf_order)
#     # docs_info = docs_vet.get_documents_info(similarity=similarity)
#
#     rdd = spark.sparkContext.parallelize(docs_info)
#
#     out = (  # this is def rdd_implementation(rdd: RDD, similarity: float) -> RDD:
#         rdd.
#         # perform mapping
#         flatMap(document_map)
#         # combine by key
#         .combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)
#         # perform reduce
#         .flatMapValues(documents_reduce)
#         # takes only pairs of doc-ids
#         .filter(lambda x: x[1][1] > similarity)
#         # get only pairs of documents
#         .map(lambda x: x[1][0])
#         # remove duplicates
#         .distinct()
#     )
#
#     t1 = time.perf_counter()
#     collected = out.collect()
#     t2 = time.perf_counter()
#
#     logging.info(f"Dataset: {data_name} - Threshold: {similarity} - idf col order: {idf_order}")
#     logging.info(f"Exec time: {t2 - t1}")
#     logging.info(f"Docs {collected}")
#
#     spark.stop()
#
#
# def get_documents_info(self, similarity: float) -> List[Tuple[str, int, List[int, float]]]:
#     """
#     Transform document vectors information in order to be processed by map-reduce framework
#     :param similarity: similarity threshold between a pair of documents
#     :return: list of triplets:
#         - doc-id, represented as a string
#         - term-threshold, referring to the index of a specific column up to which do not map terms
#         - document vector, represent the sparse vector as a list of pairs (column, value) for each non-zero entries,
#             where the column is actually a term-id
#     """
#
#     def extract_nonzero_entries(row: int) -> List[int, float]:
#         """
#         Transform the row of the matrix in list of pairs (term-id, value) for all non-zero values
#         :param row: row in the matrix corresponding to certain document
#         :return: list of tuples (term-id; value)
#         """
#
#         # extract the row
#         doc: csr_matrix = self.vectors[row]
#
#         # extract term-ids corresponding to non-zero entries
#         _, terms = doc.nonzero()
#         terms = [int(t) for t in terms]
#         terms.sort()
#
#         # extract entries corresponding to term-ids
#         entries = doc[0, terms].toarray().tolist()[0]
#
#         return list(zip(terms, entries))
#
#     # read terms_info_file depending on column order
#     # the file contains precomputed term-threshold for each document
#     terms_info_file: str = get_terms_info_idf_file(data_name=self._data_name)
#     if self._idf_order else
#     get_terms_info_file(data_name=self._data_name)
#
#     terms_info: Dict[int, Dict[float, int]] = load_terms_info(path_=terms_info_file)
#
#     # get all precomputed similarity threshold
#     sim_thresholds = list(terms_info[0].keys())
#
#     # if requested similarity was not precomputed, raise an error
#     if similarity not in sim_thresholds:
#         raise Exception(f"Terms info was not computed for {similarity} similarity, but {sim_thresholds} are available")
#
#     return [
#         (
#             self.get_row_info(row=index)[0],    # doc-id
#             terms_info[index][similarity],      # term-info
#             extract_nonzero_entries(row=index)  # list(term-id, value)
#         )
#         for index in list(range(len(self)))  # row-id
#     ]
#
#
# def get_terms_info_idf_file(data_name: str) -> str:
#     """
#     Return path to terms info idf file
#     :param data_name: name of data terms info idf vector inverse file
#     """
#
#     return path.join(get_vector_dir(data_name=data_name), f"{TERMS_INFO_IDF}.json")
