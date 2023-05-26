import os

import numpy as np
import pyspark.sql as psql
import pyspark.pandas as pds
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import types as psqlt, functions as psqlf
from pyspark.ml.feature import RegexTokenizer, IDF, CountVectorizer, CountVectorizerModel
from loguru import logger

from src.utilities.utils import AVAILABLE_CORES

ID_COL = "_id"
TITLE_COL = "title"
TEXT_COL = "text"
BEIR_CORPUS_SCHEMA: psqlt.StructType = (
    psqlt.StructType()
    .add(ID_COL, psqlt.StringType(), nullable=False)
    .add(TITLE_COL, psqlt.StringType(), nullable=False)
    .add(TEXT_COL, psqlt.StringType(), nullable=False)
)
"""
Schema of the corpus.jsonl file of a BeIR dataset
"""

COMPACTED_TEXT_COL = "compact_text"
WORDS_COL = "words"
TF_COL = "tf"

SCORES_COL = "scores"
_MIN_DOC_FREQ = 2
"""Minimum number of docs that a term must occur in to be included in the vocabulary"""


def get_document_features(
        spark: psql.SparkSession,
        corpus_json_path: str | os.PathLike,
        corpus_schema: psqlt.StructType = BEIR_CORPUS_SCHEMA,
        n_partitions: int = AVAILABLE_CORES,
        l2_normalize: bool = True
) -> psql.DataFrame:
    """
    TODO
    :param spark: current spark session
    :param corpus_json_path: path to the corpus JSON file
    :param corpus_schema: schema to load the corpus as
    :param n_partitions: number of data partitions; defaults to 1 per core
    :param l2_normalize: if True, each scores vector is divided by its l2 norm
    :return: spark dataframe with cols (ID_COL, SCORES_COL)
        (refer to the column name constants of this module)
    """

    logger.info("Loading corpus...")
    raw_docs_df: psql.DataFrame = read_corpus(
        spark=spark,
        corpus_json_path=corpus_json_path,
        corpus_schema=corpus_schema
    )

    # Make sure the number of partitions is correct
    if raw_docs_df.rdd.getNumPartitions() != n_partitions:
        raw_docs_df = raw_docs_df.repartition(numPartitions=n_partitions)

    logger.info("Compacting document texts (merging title and actual text)...")
    compact_docs_df = compact_texts(raw_docs_df=raw_docs_df)

    logger.info("Splitting texts into words...")
    with_words = split_text_into_words(compact_docs_df=compact_docs_df)

    logger.info("Calculating Scores (TF-IDF)...")
    with_tf = with_term_frequencies(with_words_df=with_words)
    with_scores_df = with_scores(with_tf_df=with_tf)

    if l2_normalize:
        logger.info("Normalizing Scores...")
        with_scores_df = normalize_scores(with_scores_df=with_scores_df)

    logger.info("Docs successfully tokenized")
    return with_scores_df


def read_corpus(
        spark: psql.SparkSession,
        corpus_json_path: str | os.PathLike,
        corpus_schema: psqlt.StructType
) -> psql.DataFrame:
    raw_docs_df: psql.DataFrame = spark.read.json(corpus_json_path, schema=corpus_schema)

    return raw_docs_df


def compact_texts(
        raw_docs_df: psql.DataFrame
) -> psql.DataFrame:
    def compact_text(df: pds.DataFrame) -> np.ndarray[str]:
        return df[TITLE_COL] + df[TEXT_COL]

    raw_docs_df: pds.DataFrame = raw_docs_df.pandas_api()

    # Create a new compacted text col
    compact_docs_df = raw_docs_df.assign(**{COMPACTED_TEXT_COL: compact_text})
    compact_docs_df = compact_docs_df.drop(columns=[TITLE_COL, TEXT_COL])
    compact_docs_df = compact_docs_df.to_spark()

    return compact_docs_df


def split_text_into_words(
        compact_docs_df: psql.DataFrame
) -> psql.DataFrame:
    regex_tokenizer = RegexTokenizer(
        inputCol=COMPACTED_TEXT_COL,
        outputCol=WORDS_COL,
        pattern=r"\W",
        gaps=True
    )
    with_words: psql.DataFrame = regex_tokenizer.transform(compact_docs_df)
    with_words = with_words.drop(COMPACTED_TEXT_COL)

    return with_words


def with_term_frequencies(
        with_words_df: psql.DataFrame
) -> psql.DataFrame:
    min_doc_freq = _MIN_DOC_FREQ  # Minimum docs a term must appear in to be included in the vocabulary
    tf_vectorizer: CountVectorizerModel = CountVectorizer(
        inputCol=WORDS_COL,
        outputCol=TF_COL,
        minDF=min_doc_freq
    ).fit(with_words_df)
    with_tf: psql.DataFrame = tf_vectorizer.transform(with_words_df)
    with_tf = with_tf.drop(WORDS_COL)

    return with_tf


def with_scores(
        with_tf_df: psql.DataFrame
) -> psql.DataFrame:
    tfidf_vectorizer = IDF(
        inputCol=TF_COL,
        outputCol=SCORES_COL,
        minDocFreq=_MIN_DOC_FREQ
    ).fit(with_tf_df)
    with_tfidf: psql.DataFrame = tfidf_vectorizer.transform(with_tf_df)
    with_tfidf = with_tfidf.drop(TF_COL)

    return with_tfidf


def normalize_scores(
        with_scores_df: psql.DataFrame
) -> psql.DataFrame:
    normalized_tf_idf_col = "normalized_tf_idf_col"

    def l2_norm(v: SparseVector) -> SparseVector:
        normalized_vals = v.values / v.norm(2)
        return SparseVector(v.size, v.indices, normalized_vals)

    l2_norm_udf = psqlf.udf(l2_norm, VectorUDT())

    normalized_tf_idf: psql.DataFrame = (
        with_scores_df
        .withColumn(
            colName=normalized_tf_idf_col,
            col=l2_norm_udf(psqlf.col(SCORES_COL))
        )
    )

    # Discard old tf_idf col and keep normalized one; keep same name for consistency
    normalized_tf_idf = normalized_tf_idf.drop(SCORES_COL)
    normalized_tf_idf = normalized_tf_idf.withColumnRenamed(existing=normalized_tf_idf_col, new=SCORES_COL)

    return normalized_tf_idf
