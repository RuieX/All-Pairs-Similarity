from typing import Sequence, Type, List, Dict, Tuple, NamedTuple
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the NLTK lemmatizer and stop words
LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))


Document: Type = Dict[str, str]
"""
{
    title: <title>
    text: <text>
}
"""
Documents: Type = Dict[str, Document]
"""
  {
    doc_id: {
      title: <title>
      text: <text>
    },
  }
"""
Queries: Type = Dict[str, str]
GroundTruth: Type = Dict[str, Dict[str, int]]
Dataset: Type = Tuple[Documents, Queries, GroundTruth]
Tokens: Type = Sequence[str]


class TokenizedText(NamedTuple):
    text_id: str
    tokens: Tokens


TokenizedDocuments: Type = List[TokenizedText]


def get_tokenized_documents(docs: Documents) -> TokenizedDocuments:
    """
    The function tokenizes each document using NLTK.
    (lowercase conversion, stopword removal, punctuation removal, lemmatization)
    :param docs:
    :return:
    """
    cleaned_documents = []

    for doc_id, doc in tqdm(docs.items(), desc="Tokenizing documents"):
        # Tokenize the document, remove punctuation and stop words, and lemmatize
        tokens = lemmatization(tokenize(compact_document(doc)))
        cleaned_doc = ' '.join(tokens)
        cleaned_documents.append(TokenizedText(text_id=doc_id, tokens=cleaned_doc))

    return cleaned_documents


def compact_document(doc: Document) -> str:
    """
    Compact form of the document
    :param doc:
    :return: string representation of the document where the title and text fields are concatenated.
    """
    return f"{doc['title']} {doc['text']}"


def tokenize(text: str) -> Tokens:
    tokens = word_tokenize(text.lower())
    tokens = remove_punctuation(remove_stopwords(tokens))
    return tokens


def remove_stopwords(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in STOPWORDS]


def remove_punctuation(tokens: Tokens) -> Tokens:
    return [t for t in tokens if t not in punctuation]


def lemmatization(tokens: Tokens) -> Tokens:
    return [LEMMATIZER.lemmatize(token) for token in tokens]
