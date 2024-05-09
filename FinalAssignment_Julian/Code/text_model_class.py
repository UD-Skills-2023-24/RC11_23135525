import ebooklib
from ebooklib import epub
import re
import os
import glob

from nltk.corpus import words
import nltk
import enchant
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

import gensim
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords, names

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

import numpy as np
from scipy.sparse import random
from sklearn.decomposition import TruncatedSVD

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

os.environ['PYENCHANT_LIBRARY_PATH'] = '/opt/homebrew/lib/libenchant-2.dylib'
import enchant

ENGLISH_DICT1 = enchant.Dict("en_UK")
ENGLISH_DICT2 = enchant.Dict("en_US")

STOP_WORDS = stopwords.words("english")
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


def is_english_word(word):
    # Initialize the Enchant English dictionary
    return (ENGLISH_DICT1.check(word) or ENGLISH_DICT2.check(word))


def preprocess(paragraphs):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    processed_doc = []

    for p in paragraphs:
        words = gensim.utils.simple_preprocess(p, min_len=3, deacc=True)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        filtered_words = [word for word in lemmatized_words if word not in stop_words]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        processed_doc.append(' '.join(stemmed_words))

    return processed_doc


def preprocess2(paragraphs):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    processed_doc = []

    for p in paragraphs:
        words = gensim.utils.simple_preprocess(p, min_len=3, deacc=True)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        stop_words = set(stopwords.words("english"))
        filtered_words = [word for word in lemmatized_words if word not in stop_words]

        processed_doc.append(' '.join(filtered_words))

    return processed_doc


def merge_strings_until_limit(strings, min_length, max_length, test_for_max=0):
    merged_string = ""
    merged_strings = []

    for s in strings:
        if len(merged_string) <= min_length:
            merged_string += s

        elif len(merged_string) > max_length and test_for_max < 5:
            splitParagraph = merged_string.split('.')
            splitParagraphRePoint = []
            for sp in splitParagraph:
                splitParagraphRePoint.append(sp + '.')

            merged = merge_strings_until_limit(splitParagraphRePoint, min_length, max_length, test_for_max + 1)
            merged_strings.extend(merged)
            merged_string = s
        else:
            merged_strings.append(merged_string)
            merged_string = s

    if merged_string:
        merged_strings.append(merged_string)

    return merged_strings


def read_epub_paragraphs(epub_file, ID, filetype):
    book = epub.read_epub(epub_file)
    paragraphs = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode('utf-8')
        content = re.sub('<[^<]+?>', '', content)
        content = re.sub('\s+', ' ', content)
        content = re.sub('\n', ' ', content)

        paragraphs.extend(content.strip().split("&#13;"))

    paragraphs = merge_strings_until_limit(paragraphs, 200, 1000)
    paragraphs = [{'paragraph': paragraphs[i], 'nr': i, 'ID': ID, 'type': filetype} for i in range(len(paragraphs))]

    return paragraphs[1:-1]



class TextModel:
    def __init__(self, files, vectorization = 'lsa', dimension = 200, epochs=10, min_df=2):
        self.vectorization = vectorization
        self.paragraphs = []

        IDs = [f.split('\\')[-1].split('.')[0] for f in files]
        for f, ID in zip(files, IDs):
            filetype = f.split('.')[-1]
            if filetype == 'epub':
                paragraph = read_epub_paragraphs(f, ID, 'epub')
                self.paragraphs.extend(paragraph)

        self.preprocessed_paragraphs = preprocess(p['paragraph'] for p in self.paragraphs)
        self.preprocessed_paragraphs2=preprocess2(p['paragraph'] for p in self.paragraphs)

        if self.vectorization == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.vector_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
        elif self.vectorization == 'lsa':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
            self.svd = TruncatedSVD(n_components=dimension, algorithm='randomized')
            self.vector_matrix = self.svd.fit_transform(self.tfidf_matrix)
        elif self.vectorization=='doc2vec':
            self.doc2vec_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.preprocessed_paragraphs2)]
            self.doc2vec_model= Doc2Vec(vector_size=dimension, min_count=1, epochs=epochs)
            self.doc2vec_model.build_vocab(self.doc2vec_documents)
            self.doc2vec_model.train(self.doc2vec_documents, total_examples=self.doc2vec_model.corpus_count,epochs= self.doc2vec_model.epochs)
            self.vector_matrix = [self.doc2vec_model.infer_vector(paragraph.split()) for paragraph in self.preprocessed_paragraphs2]

        self.nnModel = NearestNeighbors(n_neighbors=10,
                                        metric='cosine',
                                        algorithm='brute',
                                        n_jobs=-1)
        self.nnModel.fit(self.vector_matrix)


    def vectorize(self, query):
        if self.vectorization == 'lsa':
            processedQuery = preprocess([query])[0]
            tfidf_query = self.tfidf_vectorizer.transform([processedQuery])
            query_vector = self.svd.transform(tfidf_query)
            return query_vector
        elif self.vectorization == 'tfidf':
            processedQuery = preprocess([query])[0]
            query_vector = self.tfidf_vectorizer.transform([processedQuery])
            return query_vector
        elif self.vectorization=='doc2vec':
            processedQuery = preprocess([query])[0]
            doc_words = processedQuery.split()
            query_vector = self.doc2vec_model.infer_vector(doc_words)
            return query_vector


    def search(self, query, n=3, distance=False):
        if self.vectorization == 'lsa' or self.vectorization == 'tfidf':
            qv = self.vectorize(query)
            neighbours = self.nnModel.kneighbors(qv, n, return_distance=distance)[0]
            paragraphs = [self.paragraphs[i] for i in neighbours]
            return paragraphs
        elif self.vectorization=='doc2vec':
            qv = self.vectorize(query)
            neighbors = self.doc2vec_model.dv.most_similar(positive=[qv], topn=n)
            neighbors_indices = [i for i, _ in neighbors]
            paragraphs = [self.paragraphs[i] for i in neighbors_indices]
            return paragraphs


    def get_key_words(self, v, n=10):
        if self.vectorization == 'lsa':
            v = self.svd.inverse_transform(v)[0]
            top_indices = np.argpartition(v, -n)[-n:]
            words = self.tfidf_vectorizer.get_feature_names_out()
            return [words[i] for i in top_indices]
        elif self.vectorization == 'tfidf':
            top_indices = np.argpartition(v, -n)[-n:]
            words = self.tfidf_vectorizer.get_feature_names_out()
            return [words[i] for i in top_indices]



class Text_Image_MatchModel:
    def __init__(self,text_list, vectorization = 'lsa', dimension = 200, epochs=10,min_df=2):
        self.text_list=text_list
        self.vectorization = vectorization

        self.preprocessed_paragraphs = preprocess(p['text'] for p in self.text_list)
        self.preprocessed_paragraphs2 = preprocess2(p['text'] for p in self.text_list)

        if self.vectorization == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.vector_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
        elif self.vectorization == 'lsa':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
            self.svd = TruncatedSVD(n_components=dimension, algorithm='randomized')
            self.vector_matrix = self.svd.fit_transform(self.tfidf_matrix)
        elif self.vectorization == 'doc2vec':
            self.doc2vec_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.preprocessed_paragraphs2)]
            self.doc2vec_model = Doc2Vec(vector_size=dimension, min_count=1, epochs=epochs)
            self.doc2vec_model.build_vocab(self.doc2vec_documents)
            self.doc2vec_model.train(self.doc2vec_documents, total_examples=self.doc2vec_model.corpus_count,
                                     epochs=self.doc2vec_model.epochs)
            self.vector_matrix = [self.doc2vec_model.infer_vector(paragraph.split()) for paragraph in
                                  self.preprocessed_paragraphs2]

        self.nnModel = NearestNeighbors(n_neighbors=10,
                                        metric='cosine',
                                        algorithm='brute',
                                        n_jobs=-1)
        self.nnModel.fit(self.vector_matrix)

    def vectorize(self, query):
        if self.vectorization == 'lsa':
            processedQuery = preprocess([query])[0]
            tfidf_query = self.tfidf_vectorizer.transform([processedQuery])
            query_vector = self.svd.transform(tfidf_query)
            return query_vector
        elif self.vectorization == 'tfidf':
            processedQuery = preprocess([query])[0]
            query_vector = self.tfidf_vectorizer.transform([processedQuery])
            return query_vector
        elif self.vectorization == 'doc2vec':
            processedQuery = preprocess([query])[0]
            doc_words = processedQuery.split()
            query_vector = self.doc2vec_model.infer_vector(doc_words)
            return query_vector

    def search(self, query, n=3, distance=False):
        if self.vectorization == 'lsa' or self.vectorization == 'tfidf':
            qv = self.vectorize(query)
            neighbours = self.nnModel.kneighbors(qv, n, return_distance=distance)[0]
            texts = [self.text_list[i] for i in neighbours]
            return texts
        elif self.vectorization == 'doc2vec':
            qv = self.vectorize(query)
            neighbors = self.doc2vec_model.dv.most_similar(positive=[qv], topn=n)
            neighbors_indices = [i for i, _ in neighbors]
            texts = [self.text_list[i] for i in neighbors_indices]
            return texts