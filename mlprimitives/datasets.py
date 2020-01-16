# -*- coding: utf-8 -*-

"""
Datasets module.

This module contains functions that allow loading datasets for easy
testing of pipelines and primitives over multiple data modalities
and task types.

The available datasets by data modality and task type are:

+------------------+---------------+---------------------------+-------------------+
| Dataset          | Data Modality | Task Type                 | Task Subtype      |
+==================+===============+===========================+===================+
| Amazon           | Graph         | Community Detection       |                   |
+------------------+---------------+---------------------------+-------------------+
| DIC28            | Graph         | Graph Matching            |                   |
+------------------+---------------+---------------------------+-------------------+
| UMLs             | Graph         | Link Prediction           |                   |
+------------------+---------------+---------------------------+-------------------+
| Nomination       | Graph         | Vertex Nomination         |                   |
+------------------+---------------+---------------------------+-------------------+
| USPS             | Image         | Classification            | Binary            |
+------------------+---------------+---------------------------+-------------------+
| Hand Geometry    | Image         | Regression                | Univariate        |
+------------------+---------------+---------------------------+-------------------+
| Iris             | Single Table  | Classification            | Multiclass        |
+------------------+---------------+---------------------------+-------------------+
| Census           | Single Table  | Classification            | Binary            |
+------------------+---------------+---------------------------+-------------------+
| Jester           | Single Table  | Collaborative Filtering   |                   |
+------------------+---------------+---------------------------+-------------------+
| Boston           | Single Table  | Regression                | Univariate        |
+------------------+---------------+---------------------------+-------------------+
| Boston Multitask | Single Table  | Regression                | Multivariate      |
+------------------+---------------+---------------------------+-------------------+
| Wiki QA          | Multi Table   | Classification            | Binary            |
+------------------+---------------+---------------------------+-------------------+
| Personae         | Text          | Classification            | Binary            |
+------------------+---------------+---------------------------+-------------------+
| News Groups      | Text          | Classification            | Multiclass        |
+------------------+---------------+---------------------------+-------------------+
| Paper Reviews    | Text          | Regression                | Univariate        |
+------------------+---------------+---------------------------+-------------------+
"""

import io
import logging
import os
import tarfile
import urllib

import networkx as nx
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

LOGGER = logging.getLogger(__name__)

INPUT_SHAPE = [224, 224, 3]

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'data'
)
DATA_URL = 'http://mlprimitives.s3.amazonaws.com/{}.tar.gz'


class Dataset():
    """Dataset class.

    This class represents the abstraction of a dataset and works as
    a container of all the things needed in order to use a dataset
    for testing.

    Among other things, it includes the actual dataset data, information
    about its origin, a score function that works for this dataset,
    and a method to split the data in multiple ways for goodnes-of-fit
    evaluation.

    Attributes:
        name (str): Name of this dataset.
        description (str): Short description about the data that composes this dataset.
        data (array-like): Numpy array or pandas DataFrame containing all the data of
            this dataset, excluding the labels or target values.
        target (array-like): Numpy array or pandas Series containing the expected labels
            or values
        **kwargs: Any additional keyword argument passed on initailization is also
            available as instance attributes.

    Args:
        description (str): Short description about the data that composes this dataset.
            The first line of the description is expected to be a human friendly
            name for the dataset, and will be set as the `name` attribute.
        data (array-like): Numpy array or pandas DataFrame containing all the data of
            this dataset, excluding the labels or target values.
        target (array-like): Numpy array or pandas Series containing the expected labels
            or values
        score (callable): Function that will be used to compute the score of this dataset.
        shuffle (bool): Whether or not to shuffle the data before splitting.
        stratify (bool): Whther to use a stratified or regular KFold for splitting.
        **kwargs: Any additional keyword argument passed on initialization will be made
            available as instance attributes.
    """

    def __init__(self, description, data, target, score, data_modality, task_type,
                 task_subtype=None, shuffle=True, stratify=False, **kwargs):
        self.name = description.splitlines()[0]
        self.description = description

        self.data = data
        self.target = target
        self.metric = score.__name__

        self.data_modality = data_modality
        self.task_type = task_type
        self.task_subtype = task_subtype

        self._stratify = stratify
        self._shuffle = shuffle
        self._score = score

        self.extras = kwargs
        self.__dict__.update(kwargs)

    def score(self, *args, **kwargs):
        """Scoring function for this dataset.

        Args:
            \\*args, \\*\\*kwargs: Any given arguments and keyword arguments will be
            directly passed to the given scoring function.

        Returns:
            float:
                The computed score.
        """
        return self._score(*args, **kwargs)

    def __repr__(self):
        return self.name

    def describe(self):
        """Print the description of this Dataset on stdout."""
        print(self.description)
        print('Data Modality: {}'.format(self.data_modality))
        print('Task Type: {}'.format(self.task_type))
        print('Task Subtype: {}'.format(self.task_subtype))
        print('Data shape: {}'.format(self.data.shape))
        print('Target shape: {}'.format(self.target.shape))
        print('Metric: {}'.format(self.metric))
        print('Extras: {}'.format(', '.join(self.extras.keys())))

    @staticmethod
    def _get_split(data, index):
        if hasattr(data, 'iloc'):
            return data.iloc[index]
        else:
            return data[index]

    def get_splits(self, n_splits=1, random_state=0):
        """Return splits of this dataset ready for Cross Validation.

        If n_splits is 1, a tuple containing the X for train and test
        and the y for train and test is returned.
        Otherwise, if n_splits is bigger than 1, a list of such tuples
        is returned, one for each split.

        Args:
            n_splits (int): Number of times that the data needs to be splitted.

        Returns:
            tuple or list:
                if n_splits is 1, a tuple containing the X for train and test
                and the y for train and test is returned.
                Otherwise, if n_splits is bigger than 1, a list of such tuples
                is returned, one for each split.
        """
        if n_splits == 1:
            stratify = self.target if self._stratify else None

            return train_test_split(
                self.data,
                self.target,
                shuffle=self._shuffle,
                stratify=stratify,
                random_state=random_state
            )

        else:
            cv_class = StratifiedKFold if self._stratify else KFold
            cv = cv_class(n_splits=n_splits, shuffle=self._shuffle, random_state=random_state)

            splits = list()
            for train, test in cv.split(self.data, self.target):
                X_train = self._get_split(self.data, train)
                y_train = self._get_split(self.target, train)
                X_test = self._get_split(self.data, test)
                y_test = self._get_split(self.target, test)
                splits.append((X_train, X_test, y_train, y_test))

            return splits


def _download(dataset_name, dataset_path):
    url = DATA_URL.format(dataset_name)

    LOGGER.debug('Downloading dataset %s from %s', dataset_name, url)
    response = urllib.request.urlopen(url)
    bytes_io = io.BytesIO(response.read())

    LOGGER.debug('Extracting dataset into %s', DATA_PATH)
    with tarfile.open(fileobj=bytes_io, mode='r:gz') as tf:
        tf.extractall(DATA_PATH)


def _load(dataset_name):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    dataset_path = os.path.join(DATA_PATH, dataset_name)
    if not os.path.exists(dataset_path):
        _download(dataset_name, dataset_path)

    return dataset_path


def _load_images(image_dir, filenames):
    # Lazy loading of keras to avoid unnecessary keras initializations
    from keras.preprocessing.image import img_to_array, load_img   # noqa

    LOGGER.debug('Loading %s images from %s', len(filenames), image_dir)
    images = []
    for filename in filenames:
        filename = os.path.join(image_dir, filename)

        image = load_img(filename)
        image = image.resize(tuple(INPUT_SHAPE[0:2]))
        image = img_to_array(image)
        image = image / 255.0  # Quantize images.
        images.append(image)

    return np.array(images)


def _load_csv(dataset_path, name, set_index=False):
    csv_path = os.path.join(dataset_path, name + '.csv')

    LOGGER.debug('Loading csv %s', csv_path)
    df = pd.read_csv(csv_path)

    if set_index:
        df = df.set_index(df.columns[0], drop=False)

    return df


def load_usps():
    """USPs Digits dataset.

    The data of this dataset is a 3d numpy array vector with shape (224, 224, 3)
    containing 9298 224x224 RGB photos of handwritten digits, and the target is
    a 1d numpy integer array containing the label of the digit represented in
    the image.
    """
    dataset_path = _load('usps')

    df = _load_csv(dataset_path, 'data')
    X = _load_images(os.path.join(dataset_path, 'images'), df.image)
    y = df.label.values

    return Dataset(load_usps.__doc__, X, y, accuracy_score, 'image',
                   'classification', 'binary', stratify=True)


def load_handgeometry():
    """Hand Geometry dataset.

    The data of this dataset is a 3d numpy array vector with shape (224, 224, 3)
    containing 112 224x224 RGB photos of hands, and the target is a 1d numpy
    float array containing the width of the wrist in centimeters.
    """
    dataset_path = _load('handgeometry')

    df = _load_csv(dataset_path, 'data')
    X = _load_images(os.path.join(dataset_path, 'images'), df.image)
    y = df.target.values

    return Dataset(load_handgeometry.__doc__, X, y, r2_score, 'image', 'regression', 'univariate')


def load_personae():
    """Personae dataset.

    The data of this dataset is a 2d numpy array vector containing 145 entries
    that include texts written by Dutch users in Twitter, with some additional
    information about the author, and the target is a 1d numpy binary integer
    array indicating whether the author was extrovert or not.
    """
    dataset_path = _load('personae')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('label').values

    return Dataset(load_personae.__doc__, X, y, accuracy_score, 'text',
                   'classification', 'binary', stratify=True)


def load_reviews():
    """Paper Reviews Dataset.

    The data set consists of paper reviews sent to an international conference mostly in Spanish
    (some are in English). It has a total of N = 405 instances evaluated with a 5-point scale
    ('-2': very negative, '-1': negative, '0': neutral, '1': positive, '2': very positive),
    expressing the reviewer's opinion about the paper and the orientation perceived by a reader
    who does not know the reviewer's evaluation (more details in the attributes' section).
    The distribution of the original scores is more uniform in comparison to the revised scores.
    This difference is assumed to come from a discrepancy between the way the paper is evaluated
    and the way the review is written by the original reviewer.

    source: "UCI
    sourceURI: "https://archive.ics.uci.edu/ml/datasets/Paper+Reviews"
    """
    dataset_path = _load('reviews')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('evaluation').values

    return Dataset(load_reviews.__doc__, X, y, r2_score, 'text', 'regression', 'univariate')


def load_umls():
    """UMLs dataset.

    The data consists of information about a 135 Graph and the relations between
    their nodes given as a DataFrame with three columns, source, target and type,
    indicating which nodes are related and with which type of link. The target is
    a 1d numpy binary integer array indicating whether the indicated link exists
    or not.
    """
    dataset_path = _load('umls')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('label').values

    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset(load_umls.__doc__, X, y, accuracy_score, 'graph', 'link_prediction',
                   stratify=True, graph=graph)


def load_dic28():
    """DIC28 dataset from Pajek.

    This network represents connections among English words in a dictionary.
    It was generated from Knuth's dictionary. Two words are connected by an
    edge if we can reach one from the other by
    - changing a single character (e. g., work - word)
    - adding / removing a single character (e. g., ever - fever).

    There exist 52,652 words (vertices in a network) having 2 up to 8 characters
    in the dictionary. The obtained network has 89038 edges.
    """

    dataset_path = _load('dic28')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('label').values

    graph1 = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph1.gml')))
    graph2 = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph2.gml')))

    graph = graph1.copy()
    graph.add_nodes_from(graph2.nodes(data=True))
    graph.add_edges_from(graph2.edges)
    graph.add_edges_from(X[['graph1', 'graph2']].values)

    graphs = {
        'graph1': graph1,
        'graph2': graph2,
    }

    return Dataset(load_dic28.__doc__, X, y, accuracy_score, 'graph', 'graph_matching',
                   stratify=True, graph=graph, graphs=graphs)


def load_nomination():
    """Nomination dataset.

    Sample 1 of graph vertex nomination data from MIT Lincoln Lab.

    Data consists of one graph whose nodes contain two attributes, attr1 and attr2.
    Associated with each node is a label that has to be learned and predicted.
    """

    dataset_path = _load('nomination')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('label').values

    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset(load_nomination.__doc__, X, y, accuracy_score, 'graph', 'vertex_nomination',
                   stratify=True, graph=graph)


def load_amazon():
    """Amazon dataset.

    Amazon product co-purchasing network and ground-truth communities.

    Network was collected by crawling Amazon website. It is based on Customers Who Bought
    This Item Also Bought feature of the Amazon website. If a product i is frequently
    co-purchased with product j, the graph contains an undirected edge from i to j.
    Each product category provided by Amazon defines each ground-truth community.
    """

    dataset_path = _load('amazon')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('label').values

    graph = nx.Graph(nx.read_gml(os.path.join(dataset_path, 'graph.gml')))

    return Dataset(load_amazon.__doc__, X, y, normalized_mutual_info_score, 'graph',
                   'community_detection', graph=graph)


def load_jester():
    """Jester dataset.

    Ratings from the Jester Online Joke Recommender System.

    This dataset consists of over 1.7 million instances of (user_id, item_id, rating)
    triples, which is split 50-50 into train and test data.

    source: "University of California Berkeley, CA"
    sourceURI: "http://eigentaste.berkeley.edu/dataset/"
    """

    dataset_path = _load('jester')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('rating').values

    return Dataset(load_jester.__doc__, X, y, r2_score, 'single_table', 'collaborative_filtering')


def load_census():
    """Adult Census dataset.

    Predict whether income exceeds $50K/yr based on census data. Also known as "Adult" dataset.

    Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean
    records was extracted using the following conditions: ((AAGE>16) && (AGI>100) &&
    (AFNLWGT>1)&& (HRSWK>0))

    Prediction task is to determine whether a person makes over 50K a year.

    source: "UCI
    sourceURI: "https://archive.ics.uci.edu/ml/datasets/census+income"
    """

    dataset_path = _load('census')

    X = _load_csv(dataset_path, 'data')
    y = X.pop('income').values

    return Dataset(load_census.__doc__, X, y, accuracy_score, 'single_table',
                   'classification', 'binary', stratify=True)


def load_wikiqa():
    """WikiQA dataset.

    A Challenge Dataset for Open-Domain Question Answering.

    WikiQA dataset is a publicly available set of question and sentence (QS) pairs,
    collected and annotated for research on open-domain question answering.

    source: "Microsoft"
    sourceURI: "https://www.microsoft.com/en-us/research/publication/wikiqa-a-challenge-dataset-for-open-domain-question-answering/#"
    """  # noqa

    dataset_path = _load('wikiqa')

    data = _load_csv(dataset_path, 'data', set_index=True)
    questions = _load_csv(dataset_path, 'questions', set_index=True)
    sentences = _load_csv(dataset_path, 'sentences', set_index=True)
    vocabulary = _load_csv(dataset_path, 'vocabulary', set_index=True)

    entities = {
        'data': (data, 'd3mIndex', None),
        'questions': (questions, 'qIndex', None),
        'sentences': (sentences, 'sIndex', None),
        'vocabulary': (vocabulary, 'index', None)
    }
    relationships = [
        ('questions', 'qIndex', 'data', 'qIndex'),
        ('sentences', 'sIndex', 'data', 'sIndex')
    ]

    target = data.pop('isAnswer').values

    return Dataset(load_wikiqa.__doc__, data, target, accuracy_score, 'multi_table',
                   'classification', 'binary', startify=True,
                   entities=entities, relationships=relationships)


def load_newsgroups():
    """20 News Groups dataset.

    The data of this dataset is a 1d numpy array vector containing the texts
    from 11314 newsgroups posts, and the target is a 1d numpy integer array
    containing the label of one of the 20 topics that they are about.
    """
    dataset = datasets.fetch_20newsgroups()
    return Dataset(load_newsgroups.__doc__, np.array(dataset.data), dataset.target,
                   accuracy_score, 'text', 'classification', 'multiclass', stratify=True)


def load_iris():
    """Iris dataset."""
    dataset = datasets.load_iris()
    return Dataset(load_iris.__doc__, dataset.data, dataset.target,
                   accuracy_score, 'single_table', 'classification',
                   'multiclass', stratify=True)


def load_boston():
    """Boston House Prices dataset."""
    dataset = datasets.load_boston()
    return Dataset(load_boston.__doc__, dataset.data, dataset.target, r2_score,
                   'single_table', 'regression', 'univariate')


def load_boston_multitask():
    """Boston House Prices dataset.

    Modified version of the Boston dataset with a synthetic multitask output.

    The multitask output is obtained by applying a linear transformation
    to the original y and adding it as a second output column.
    """
    dataset = datasets.load_boston()
    y = dataset.target
    target = np.column_stack([y, 2 * y + 5])
    return Dataset(load_boston.__doc__, dataset.data, target, r2_score,
                   'single_table', 'regression', 'multivariate')


def load_dataset(name):
    loader_name = 'load_' + name
    loader = globals().get(loader_name)

    if not loader:
        raise ValueError('Unknown dataset: "{}"'.format(name))

    return loader()
