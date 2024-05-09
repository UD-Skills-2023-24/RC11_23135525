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

import moviepy
from moviepy.editor import *
from moviepy import *
import shutil
import numpy as np
from datetime import datetime
from PIL import Image
import subprocess
from IPython.display import display, Video
from moviepy.editor import VideoFileClip
from collections import defaultdict

from moviepy.editor import VideoFileClip
from IPython.display import display, Video
import subprocess
from IPython.display import Video

def loadVideosToInfile(folder):
    videos = os.listdir(folder)
    with open('input', 'w') as inputfile:
        for v in videos:
            if v.split('.')[-1] == 'mp4':
                inputfile.write(v+'\n')


def remove_spaces_in_filenames(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            new_file_name = file.replace(' ', '')
            if new_file_name != file:
                new_file_path = os.path.join(root, new_file_name)
                os.rename(file_path, new_file_path)
                print(f"file {file} the empty space is removed,the new file name is {new_file_name}")


def parse_srt_file(file_path):
    subtitles = []
    current_subtitle = None

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()

            if line.isdigit():
                if current_subtitle:
                    subtitles.append(current_subtitle)
                current_subtitle = {'index': int(line)}
            elif ' --> ' in line:
                start_time, end_time = line.split(' --> ')
                if current_subtitle:
                    current_subtitle.update({
                        'start_time': start_time.strip(),
                        'end_time': end_time.strip(),
                        'text': []
                    })
            elif line:
                if current_subtitle is not None:
                    current_subtitle.setdefault('text', []).append(line)

    if current_subtitle:
        subtitles.append(current_subtitle)

    return subtitles


def add_filename_to_subtitles(subtitles, file_name):
    for subtitle in subtitles:
        subtitle['file_name'] = file_name
    return subtitles


def process_srt_files(folder_path):
    subtitles_list = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.srt'):
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                try:
                    subtitles = parse_srt_file(file_path)
                    subtitles_with_filename = add_filename_to_subtitles(subtitles, file_name)
                    subtitles_list.extend(subtitles_with_filename)
                except Exception as e:
                    print(f"Error processing file: {file_path}")
                    print(f"Error message: {e}")

    return subtitles_list

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


def is_english_word(word):
    return (ENGLISH_DICT1.check(word) or ENGLISH_DICT2.check(word))


def merge_subtitles(subtitles):
    merged_subtitles = defaultdict(list)
    for subtitle in subtitles:
        if 'text' in subtitle:
            merged_subtitles[subtitle['file_name']].append(subtitle)

    result = []
    for file_name, subtitles_list in merged_subtitles.items():
        for i in range(0, len(subtitles_list), 3):
            block_texts = []
            for sub in subtitles_list[i:i+3]:
                if isinstance(sub['text'], list):
                    block_texts.extend(sub['text'])
                else:
                    block_texts.append(sub['text'])

            block_start_time = subtitles_list[i]['start_time']
            block_end_time = subtitles_list[min(i+2, len(subtitles_list)-1)]['end_time']
            merged_text = ' '.join(block_texts)

            result.append({
                'index': len(result) + 1,
                'start_time': block_start_time,
                'end_time': block_end_time,
                'text': merged_text,
                'file_name': file_name
            })

    return result


def check_film_exist(result, folder_path):
    if result:
        matched_films = result[0]['file_name']
        print(matched_films)
        mp4_name = f"{matched_films}.mp4"
        mp4_path = os.path.join(folder_path, mp4_name)

        if os.path.exists(mp4_path):
            print(f"Found: {mp4_path}")
        else:
            print(f"No matching: {mp4_name}")
    return mp4_path


def display_matched_filmclip(result, movie_path,query):
    start_time_parts = result[0]['start_time'].split(',')
    start_seconds = int(start_time_parts[0].split(':')[-1]) + 60 * int(start_time_parts[0].split(':')[1]) + 3600 * int(start_time_parts[0].split(':')[0])
    end_time_parts = result[0]['end_time'].split(',')
    end_seconds = int(end_time_parts[0].split(':')[-1]) + 60 * int(end_time_parts[0].split(':')[1]) + 3600 * int(end_time_parts[0].split(':')[0])

    #output_folder=r'E:\Document\Dylan Zhang File\UCL\Term2\Skill_Class\assignment\Julian\Datasets\output_clips'
    #filename= 'matched_clip_{}.mp4'.format(query)
    #output_path = os.path.join(output_folder, filename)
    output_path='matched_clip_{}.mp4'.format(query)

    ffmpeg_command = ['ffmpeg', '-i', movie_path, '-ss', str(start_seconds), '-to', str(end_seconds), '-c', 'copy', output_path]
    subprocess.run(ffmpeg_command, capture_output=True, text=True)

    display(Video(output_path, width=800,embed=True))


def display_matched_frame(movie_path,frame_name,start_time,end_time):

    output_path = 'frame_clip_{}.mp4'.format(frame_name)

    ffmpeg_command = ['ffmpeg', '-i', movie_path, '-ss', str(start_time), '-to', str(end_time), '-c', 'copy',output_path]

    subprocess.run(ffmpeg_command, capture_output=True, text=True)

    display(Video(output_path, width=800,embed=True))


class FilmSubtitleModel:
    def __init__(self,files,vectorization = 'lsa', dimension = 200, epochs=10, min_df=2):
        self.vectorization = vectorization
        self.subtitles = files

        self.preprocessed_subtitles = preprocess(p['text'] for p in self.subtitles)
        self.preprocessed_subtitles2 = preprocess2(p['text'] for p in self.subtitles)

        if self.vectorization == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.vector_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_subtitles)
        elif self.vectorization == 'lsa':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_subtitles)
            self.svd = TruncatedSVD(n_components=dimension, algorithm='randomized')
            self.vector_matrix = self.svd.fit_transform(self.tfidf_matrix)
        elif self.vectorization=='doc2vec':
            self.doc2vec_documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.preprocessed_subtitles2)]
            self.doc2vec_model= Doc2Vec(vector_size=dimension, min_count=1, epochs=epochs)
            self.doc2vec_model.build_vocab(self.doc2vec_documents)
            self.doc2vec_model.train(self.doc2vec_documents, total_examples=self.doc2vec_model.corpus_count,epochs= self.doc2vec_model.epochs)
            self.vector_matrix = [self.doc2vec_model.infer_vector(paragraph.split()) for paragraph in self.preprocessed_subtitles2]

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
            processedQuery = preprocess2([query])[0]
            doc_words = processedQuery.split()
            query_vector = self.doc2vec_model.infer_vector(doc_words)
            return query_vector


    def search(self, query, n=3, distance=False):
        if self.vectorization == 'lsa' or self.vectorization == 'tfidf':
            qv = self.vectorize(query)
            neighbours = self.nnModel.kneighbors(qv, n, return_distance=distance)[0]
            paragraphs = [self.subtitles[i] for i in neighbours]
            return paragraphs
        elif self.vectorization=='doc2vec':
            qv = self.vectorize(query)
            neighbors = self.doc2vec_model.dv.most_similar(positive=[qv], topn=n)
            neighbors_indices = [i for i, _ in neighbors]
            subtitles = [self.subtitles[i] for i in neighbors_indices]
            return subtitles

