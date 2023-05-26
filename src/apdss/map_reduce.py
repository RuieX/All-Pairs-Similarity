import itertools
import operator
import time
from typing import Tuple, Type, Sequence, Iterable, Mapping, List

import numpy as np
import pyspark as ps
import pyspark.sql as psql
from loguru import logger
from pyspark.ml.linalg import SparseVector
from pyspark.sql import types as psqlt

import src.tokenization as tok
from src.apdss.core import DocId, Document, DocsSimilarity, APDSS, cosine_sim, Results
from src.utilities.utils import AVAILABLE_CORES


TermId: Type = int

TermScorePair: Type = Tuple[TermId, float]
"""
(term_id, score) pair
"""

DenseScores: Type = np.ndarray[float]

SortingOrder: Type = np.ndarray[int]

SortedTermIds: Type = np.ndarray[TermId]

SortedDStar: Type = Tuple[SortedTermIds, DenseScores]
"""
Pair of arrays representing respectively the term_ids and scores,
both sorted by the scores, that d* consists of

Note: this means that sorted_term_ids[i] is the actual term_id of sorted_scores[i]

Note: d* is the "maximum document", i.e. a document such that d[i] = max d[i] for any d in corpus
"""

BPair: Type = Tuple[DocId, int]
"""
Return type of the b(document, d*) function
"""

BMap: Type = Mapping[DocId, int]
"""
Map that pairs each document id with the term_id obtained by applying 
b(document, d*) function
"""

TermDocumentPair: Type = Tuple[DocId, Document]

TermDocumentPairsGroup: Type = Tuple[TermId, Iterable[TermDocumentPair]]


class MapReduceAPDSS(APDSS):
    def apdss(
            self,
            spark: psql.SparkSession,
            docs_scores_df: psql.DataFrame,
            threshold: float,
            num_partitions: int = AVAILABLE_CORES
    ) -> Results:
        start_time = time.time()

        # Cache input df and un-persist at the end of the func to avoid side effects
        # Caching this is important because it is involved in a lot of transformations
        is_input_df_cached = docs_scores_df.is_cached
        if not is_input_df_cached:
            logger.debug(f"Caching input df")
            docs_scores_df = docs_scores_df.cache()

        d_star: SortedDStar = _get_d_star(docs_scores_df)
        d_star_broadcast: ps.Broadcast[SortedDStar] = spark.sparkContext.broadcast(d_star)

        sorting_order = _get_sorting_order(d_star[0])
        sorting_order_broadcast = spark.sparkContext.broadcast(sorting_order)

        b_map: BMap = _get_b_map(
            docs_scores_df=docs_scores_df,
            d_star_broadcast=d_star_broadcast,
            threshold=threshold,
            sorting_order_broadcast=sorting_order_broadcast
        )
        b_map_broadcast: ps.Broadcast[BMap] = spark.sparkContext.broadcast(b_map)

        # Map dataframe rows to documents tuple (doc_id, doc_scores)
        documents_rdd: ps.RDD[Document] = docs_scores_df.rdd.map(lambda row: (row[tok.ID_COL], row[tok.SCORES_COL]))

        # For each document, create one or more (term_id, document) tuples by applying prefix filtering
        term_doc_pairs_rdd: ps.RDD[TermDocumentPair] = documents_rdd.flatMap(
            f=lambda doc: _apply_prefix_filtering(
                doc=doc,
                b_map=b_map_broadcast.value,
                sorted_term_ids=d_star_broadcast.value[0],
                sorting_order=sorting_order_broadcast.value
            )
        )

        logger.debug(f"{_apply_prefix_filtering.__name__} partitions {term_doc_pairs_rdd.getNumPartitions()}")

        # For each key, group all elements into a single collection
        # Efficiency Note: as per this SO answer https://stackoverflow.com/a/39316189/19582401
        #   it is suggested to use groupByKey as opposed to reduceByKey or combineByKey, because
        #   they both result in a huge number (O(n), O(n^2)) of list instantiations
        term_doc_sequence_rdd: ps.RDD[TermDocumentPairsGroup] = term_doc_pairs_rdd.groupByKey(
            numPartitions=num_partitions
        )

        logger.debug(f"term_doc_sequence_rdd partitions {term_doc_pairs_rdd.getNumPartitions()}")

        # Perform similarity search on document grouped under the same term_id (key)
        similar_docs_pairs_rdd = term_doc_sequence_rdd.flatMap(
            f=lambda pair: _get_similar_docs_pairs(
                term_docs_pair=pair,
                threshold=threshold,
            )
        )

        similar_docs: Sequence[DocsSimilarity] = similar_docs_pairs_rdd.collect()

        end_time = time.time()

        if is_input_df_cached:
            logger.debug(f"Un-persisting input df")
            docs_scores_df.unpersist()

        return Results(
            time=end_time - start_time,
            similar_docs=similar_docs
        )


def _get_d_star(docs_scores_df: psql.DataFrame) -> SortedDStar:
    scores_only = docs_scores_df.select(tok.SCORES_COL)
    scores_rdd: ps.RDD[psqlt.Row] = scores_only.rdd

    logger.debug(f"{_get_d_star.__name__} partitions {scores_rdd.getNumPartitions()}")

    def to_term_scores_pairs(row: psqlt.Row):
        doc_scores: SparseVector = row[tok.SCORES_COL]
        for i, term_id in enumerate(doc_scores.indices):
            yield term_id, doc_scores.values[i]

    # Take the maximum score for each term_id
    d_star_rdd: ps.RDD[TermScorePair] = scores_rdd.flatMap(f=to_term_scores_pairs).reduceByKey(max)

    # Sort descending by tf_idf
    d_star_rdd = d_star_rdd.sortBy(keyfunc=operator.itemgetter(1), ascending=False)

    # Group all the (term_id, tf-idf) pairs into two separate lists
    d_star_pairs: Sequence[TermScorePair] = d_star_rdd.collect()
    sorted_term_ids, d_star = zip(*d_star_pairs)

    return np.array(sorted_term_ids), np.array(d_star)


def _get_b_map(
        docs_scores_df: psql.DataFrame,
        d_star_broadcast: ps.Broadcast[SortedDStar],
        threshold: float,
        sorting_order_broadcast: ps.Broadcast[SortingOrder]
) -> BMap:
    doc_scores_rdd: ps.RDD[psqlt.Row] = docs_scores_df.rdd

    logger.debug(f"{_get_b_map.__name__} partitions {doc_scores_rdd.getNumPartitions()}")

    def apply_b(doc: psqlt.Row) -> BPair:
        doc: Document = doc[tok.ID_COL], doc[tok.SCORES_COL]

        return _b(
            doc=doc,
            d_star=d_star_broadcast.value,
            threshold=threshold,
            sorting_order=sorting_order_broadcast.value
        )

    b_pairs: ps.RDD[BPair] = doc_scores_rdd.map(apply_b)
    b_map: BMap = b_pairs.collectAsMap()

    return b_map


def _b(
        doc: Document,
        d_star: SortedDStar,
        threshold: float,
        sorting_order: SortingOrder = None
) -> BPair:
    """
    Returns a pair (doc_id, t), where `t` is defined as the largest index `i` at which
    the cumulative sum at `i` of the element-wise product between `doc` and `d*`
    is less than the provided threshold

    :param doc: document to apply the function to
    :param d_star: "maximum document", along with the collection of sorted term ids
    :param threshold: similarity threshold
    :param sorting_order: map M such that M[term_id] = i
        and sorted_term_ids[i] = term_id
    :return: pair (doc_id, t)
    """

    doc_id, doc_scores = doc
    sorted_term_ids, d_star_scores = d_star

    if sorting_order is None:
        sorting_order = _get_sorting_order(sorted_term_ids)

    dense_doc_scores = doc_scores.toArray()
    sorted_dense_doc_scores = dense_doc_scores[sorting_order]
    element_wise_prod: np.ndarray = np.multiply(sorted_dense_doc_scores, d_star_scores)

    cum_sum: np.ndarray = np.cumsum(element_wise_prod)

    # The cumulative-sum array is sorted by definition,
    #   since every score (and therefore the prod. of scores) is >= 0;
    #   np.searchsorted can therefore be performed
    # Additionally, recall that `t` is defined as the largest index at which
    #   the above cum-sum is less than the threshold
    t = np.searchsorted(a=cum_sum, v=threshold, side="left")

    return doc_id, t


def _get_sorting_order(sorted_term_ids: SortedTermIds) -> SortingOrder:
    """
    Returns a mapping M such that M[term_id] = i, where sorted_term_ids[i] = term_id
    :param sorted_term_ids: array containing term ids sorted by some criteria
    :return: arg-sort mapping
    """

    sorting_order = np.zeros(shape=sorted_term_ids.size, dtype=int)
    sorting_order[sorted_term_ids] = np.arange(start=0, stop=sorted_term_ids.size, step=1, dtype=int)

    return sorting_order


def _apply_prefix_filtering(
        doc: Document,
        sorted_term_ids: SortedTermIds,
        b_map: BMap,
        sorting_order: SortingOrder = None
) -> Sequence[TermDocumentPair]:
    """
    Maps a document to term-document pairs by applying prefix-filtering

    :param doc: document to map
    :param sorted_term_ids: sorted term ids, used to
    :param b_map:
    :param sorting_order: map M such that M[term_id] = i
        and sorted_term_ids[i] = term_id
    :return: sequence of term-document pairs
    """

    doc_id, sparse_scores = doc

    # Map M such that M[term_id] = i and sorted_term_ids[i] = term_id
    if sorting_order is None:
        sorting_order = _get_sorting_order(sorted_term_ids)

    pairs: List[TermDocumentPair] = []
    for term_id in sparse_scores.indices:
        t = b_map[doc_id]
        if sorting_order[term_id] > t:
            pairs.append((term_id, doc))

    return pairs


def _get_similar_docs_pairs(
        term_docs_pair: Tuple[TermId, Sequence[Document]],
        threshold: float,
        assume_scores_normalized: bool = True
) -> Sequence[DocsSimilarity]:
    """
    :param term_docs_pair: term_id paired with all the documents mapped to that term
    :param threshold: similarity threshold
    :param assume_scores_normalized: if True, the scores are assumed to be normalized,
        i.e. divided by their l2 norm, which helps to speed up calculations
    :return: collection of similar document pairs and similarity tuples
    """

    term_id, docs = term_docs_pair

    similar_docs: List[DocsSimilarity] = []
    all_doc_pairs: Iterable[Tuple[Document, Document]] = itertools.combinations(docs, r=2)
    for doc_a, doc_b in all_doc_pairs:
        id_a, scores_a = doc_a
        id_b, scores_b = doc_b

        # Intersection is surely != empty_set because all the docs in the provided collection
        #   have at least the associated `term_id` in common
        # Indices in both document scores are unique
        common_terms: np.ndarray = np.intersect1d(scores_a.indices, scores_b.indices, assume_unique=True)

        # The goal of this check is ensuring that only one reducer,
        #   the one that received the largest common term_id, actually does the calculations
        # This prevents multiple copies of the same result and is more efficient
        if term_id == max(common_terms):
            if assume_scores_normalized:
                # Dot product: already divided by l2 norm
                sim_score = scores_a.dot(scores_b)
            else:
                # Cosine similarity: divide dot product by product of l2 norms
                sim_score = cosine_sim(scores_a, scores_b)

            if sim_score >= threshold:
                docs_similarity: DocsSimilarity = (id_a, id_b, sim_score)
                similar_docs.append(docs_similarity)

    return similar_docs
