from sklearn.cluster import KMeans
import random
import scipy
import sklearn
import tensorflow as tf
import PIL
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dropout, Dense, Softmax)
from tensorflow.keras.applications import mobilenet as _mobilenet
import random
import os
import re
import numpy as np
from PIL import Image
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from IPython.display import Image as iImage
from IPython.display import display
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics.pairwise import cosine_similarity


def normalise(train, p):
    min_d = np.min(train)
    max_d = np.max(train)
    normalised_p = (p-min_d)/(max_d - min_d)
    return normalised_p

def denormalise(train, p):
    min_d = np.min(train)
    max_d = np.max(train)
    denormalised_p = p * (max_d - min_d) + min_d
    return denormalised_p

# Main routine for training an SOM. It requires an initialized SOM grid
# or a partially trained grid as parameter
def train_SOM(SOM, train_data, learn_rate=.1, radius_sq=1,
                    lr_decay=.1, radius_decay=.1, epochs=10):
    rand = np.random.RandomState(0)
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)

        for i, train_ex in enumerate(train_data):
            g, h = find_BMU1(SOM, train_ex)
            SOM = update_weights(SOM, train_ex,
                                 learn_rate, radius_sq, (g, h))
            print(f"Epoch {epoch + 1}/{epochs}, Sample {i + 1}/{len(train_data)}", end="\r")

        # Update learning rate and radius
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)
    return SOM

def cosine(a,b):
    return cosine_similarity([a], [b])[0][0]

def u_matrix1(lattice):
    """Builds a U-matrix on top of the trained lattice.

        Parameters
        ---
        lattice : list

            The SOM generated lattice

        Returns
        ---
        The lattice of the shape (R,C):

        R - number of rows; C - number of columns;
        """
    X, Y, Z = lattice.shape
    u_values = np.empty((X,Y), dtype=np.float64)
    for y in range(Y):
        for x in range(X):
            current = lattice[x,y]
            dist = 0
            num_neigh = 0
            # left
            if x-1 >= 0:
                #middle
                vec = lattice[x-1,y]
                dist += cosine(current, vec)
                num_neigh += 1
                if y - 1 >= 0:
                    #sup
                    vec = lattice[x-1, y-1]
                    dist += cosine(current, vec)
                    num_neigh += 1
                if y + 1 < Y:
                    # down
                    vec = lattice[x-1,y+1]
                    dist += cosine(current, vec)
                    num_neigh += 1
            # middle
            if y - 1 >= 0:
                # up
                vec = lattice[x,y-1]
                dist += cosine(current, vec)
                num_neigh += 1
            # down
            if y + 1 < Y:
                vec = lattice[x,y+1]
                dist += cosine(current, vec)
                num_neigh += 1
            # right
            if x + 1 < X:
                # middle
                vec = lattice[x+1,y]
                dist += cosine(current, vec)
                num_neigh += 1
                if y - 1 >= 0:
                    #up
                    vec = lattice[x+1,y-1]
                    dist += cosine(current, vec)
                    num_neigh += 1
                if y + 1 < lattice.shape[1]:
                    # down
                    vec = lattice[x+1,y+1]
                    dist += cosine(current, vec)
                    num_neigh += 1
            u_values[x,y] = dist / num_neigh
    u_values = (u_values - 1) * -1
    return u_values


def u_matrix(lattice):
    """Builds a U-matrix on top of the trained lattice.

        Parameters
        ---
        lattice : list

            The SOM generated lattice

        Returns
        ---
        The lattice of the shape (R,C):

        R - number of rows; C - number of columns;
        """
    X, Y, Z = lattice.shape
    u_values = np.empty((X, Y), dtype=np.float64)

    for y in range(Y):
        for x in range(X):
            current = lattice[x, y]
            dist = 0
            num_neigh = 0
            # left
            if x - 1 >= 0:
                # middle
                vec = lattice[x - 1, y]
                dist += euclidean(current, vec)
                num_neigh += 1
                if y - 1 >= 0:
                    # sup
                    vec = lattice[x - 1, y - 1]
                    dist += euclidean(current, vec)
                    num_neigh += 1
                if y + 1 < Y:
                    # down
                    vec = lattice[x - 1, y + 1]
                    dist += euclidean(current, vec)
                    num_neigh += 1
            # middle
            if y - 1 >= 0:
                # up
                vec = lattice[x, y - 1]
                dist += euclidean(current, vec)
                num_neigh += 1
            # down
            if y + 1 < Y:
                vec = lattice[x, y + 1]
                dist += euclidean(current, vec)
                num_neigh += 1
            # right
            if x + 1 < X:
                # middle
                vec = lattice[x + 1, y]
                dist += euclidean(current, vec)
                num_neigh += 1
                if y - 1 >= 0:
                    # up
                    vec = lattice[x + 1, y - 1]
                    dist += euclidean(current, vec)
                    num_neigh += 1
                if y + 1 < lattice.shape[1]:
                    # down
                    vec = lattice[x + 1, y + 1]
                    dist += euclidean(current, vec)
                    num_neigh += 1
            u_values[x, y] = dist / num_neigh
    u_values = (u_values - 1) * -1
    return u_values

def calculateQE(SOM, data):
        sumSqDist = 0
        for d in data:
            g, h = find_BMU1(SOM, d)
            v1 = SOM[g, h]
            v2 = d
            sumSqDist += scipy.spatial.distance.cdist([v1], [v2], 'cosine')[0][0]
        QE = sumSqDist / len(data)
        return QE

def calculateTE(SOM, data):
        failed = 0
        for d in data:
            g1, h1 = find_BMU1(SOM, d)
            g2, h2 = find_BMU_2(SOM, d)
            dist = scipy.spatial.distance.cityblock([g1, h1], [g2, h2])
            if dist > 1:
                failed += 1
        return failed / len(data)


# Return the (g,h) index of the BMU in the grid
def find_BMU_2(SOM,x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argpartition(distSq, 2, axis=None)[2], distSq.shape)

def euclidean(a, b):
    return np.linalg.norm(a-b)

def activate(SOM, p):
    activatedSOM = np.array([[euclidean(p, c) for c in r] for r in SOM])
    return activatedSOM

def activate1(train_data, SOM, p):
    #normalP = normalise(train_data, p)
    activatedSOM = np.array([[cosine_similarity([p], [c])[0][0] for c in r] for r in SOM])
    normalisedActivatedSOM = normalise(activatedSOM, activatedSOM)
    #activatedSOM = (normalisedActivatedSOM -1)*(-1)
    return activatedSOM

# Return the (g,h) index of the BMU in the grid
def find_BMU(SOM,x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

def find_BMU1(SOM, x):
    simSOM = SOM.reshape((-1, len(x)))
    cos_sims = cosine_similarity([x], simSOM)
    return np.unravel_index(np.argmax(cos_sims, axis=None), SOM.shape[:2])

# Update the weights of the SOM cells when given a single training example
# and the model parameters along with BMU coordinates as a tuple
def update_weights(SOM, train_ex, learn_rate, radius_sq,
                   BMU_coord, step=3):
    g, h = BMU_coord
    #if radius is close to zero then only BMU is changed
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    # Change all cells in a small neighborhood of BMU
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])
    return SOM

def normalise_2(train):
    min_d = np.min(train, axis=0)
    max_d = np.max(train, axis=0)
    normalised_train = (train - min_d) / (max_d - min_d)
    return normalised_train

def find_closest_cos(data, v):
    cos_dist = cosine_similarity(data, [v])
    return np.argmax(cos_dist)


class SOMModel:
    def __init__(self, learn_rates=[0.3, 0.5, 0.6], radius_sqs=[1, 5, 7], m=10, n=10, pca_result=None, vector_size=4):
        self.SOMS = []
        self.rate = learn_rates
        self.radius = radius_sqs
        self.size = vector_size

        for learn_rate in self.rate:
            for radius_sq in self.radius:
                rand = np.random.RandomState(0)
                SOM = rand.uniform(0, 1, (m, n, vector_size))
                SOM = train_SOM(SOM, pca_result, epochs=20, learn_rate=learn_rate, radius_sq=radius_sq)
                self.SOMS.append(SOM)

        fig, axes = plt.subplots(
            nrows=3, ncols=3, figsize=(10, 10),
            subplot_kw=dict(xticks=[], yticks=[]))
        param_combinations = [(lr, rsq) for lr in self.rate for rsq in self.radius]

        for s, ax, (learn_rate, radius_sq) in zip(self.SOMS, axes.flat, param_combinations):
            u_matrix_values = u_matrix(s)
            custom_cmap = plt.cm.viridis
            im = ax.imshow(u_matrix_values, cmap=custom_cmap, aspect='auto')
            QE = round(calculateQE(s, pca_result), 4)
            TE = round(calculateTE(s, pca_result), 4)
            ax.set_title('$\eta$ = ' + str(learn_rate) +
                         ', $\sigma^2$ = ' + str(radius_sq) +
                         ', \n$QE$ = ' + str(QE) +
                         ', $TE$ = ' + str(TE))

        plt.tight_layout()
        cbar = plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label('Distance')
        plt.show()

    def SOM_UMatrix(self, SOM, pca_result, learn_rate, radius_sq, epochs, custom_cmap=plt.cm.viridis):
        self.trained_SOM = train_SOM(SOM, pca_result,learn_rate=learn_rate, radius_sq=radius_sq, epochs=epochs)

        self.learn_rate= learn_rate
        self.radius_sq= radius_sq

        self.u_matrix_values = u_matrix(self.trained_SOM)
        self.QE = round(calculateQE(self.trained_SOM, pca_result), 4)
        self.TE = round(calculateTE(self.trained_SOM, pca_result), 4)

        fig, ax = plt.subplots(figsize=(15, 15))
        colors = [
            (0, "#FFD700"),  # 浅橙色
            (0.4, "#FFFFE0"),  # 浅黄色
            (0.8, "#D3D3D3"),  # 浅灰色
            (1, "#FFFFFF")  # 白色
        ]

        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        im = plt.imshow(self.u_matrix_values, cmap=cmap, aspect='auto')
        ax.set_title('U-Matrix' + '\n$\eta$ = ' + str(self.learn_rate) +
                     ', $\sigma^2$ = ' + str(self.radius_sq) +
                     ', \n$QE$ = ' + str(self.QE) +
                     ', $TE$ = ' + str(self.TE))

        plt.colorbar(im, ax=ax, shrink=0.95, label='Distance')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()


    def Text_UMatrix(self, neuron_word_mapping,name=None,fontsize=2, dpi=500):
        plt.figure(figsize=(20, 20))
        colors = [
            (0, "#FFD700"),  # 浅橙色
            (0.4, "#FFFFE0"),  # 浅黄色
            (0.8, "#D3D3D3"),  # 浅灰色
            (1, "#FFFFFF")  # 白色
        ]

        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        im = plt.imshow(self.u_matrix_values, cmap=cmap, aspect='auto')
        plt.title('U-Matrix' + '\nemoji='+str(name)+', $\eta$ = ' + str(self.learn_rate) +
                  ', $\sigma^2$ = ' + str(self.radius_sq) +
                  ', \n$QE$ = ' + str(self.QE) +
                  ', $TE$ = ' + str(self.TE))
        plt.colorbar(im, shrink=0.95, label='Distance',pad=0.01)

        for (i, j), sentence in neuron_word_mapping.items():
            if not sentence:
                continue
            words=sentence[0].split(' ')[:8]
            if len(words)>=8:
                text=(words[0]+'\n'+words[1]+''+words[2]+'\n'+words[3]+' '+words[4]+
                      '\n'+words[5]+''+words[6]+'\n'+words[7])
            else:
                text=''.join(words)

            plt.text(j, i, text, ha='center', va='center', color='black', fontsize=fontsize)

        plt.xticks([])
        plt.yticks([])
        plt.show()
