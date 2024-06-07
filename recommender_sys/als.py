"""
Includes classes for the Alternating Least Squares (ALS) method to solve the
problem of collaborative filtering in recommender systems. I am designing the
implementations in such a way that the most used settings (similar to pyspark.als)
will be used. There are two versions of the implementation, one uses Numpy and the
other uses Tensorflow. Comparison of the results with pyspark implementation is
stored in evaluation.als.

The first problem encountered is the storage of sparse matrices. Creating a new class
to handle them in Numpy
"""
import numpy as np
from scipy.sparse import csr_matrix


class ALSNum:
    def __init__(self, ratings: csr_matrix, factors: int, regularization: float):
        """Alternating Least Squares - NumPy Variant

        Implements the alternating least squares algorithm to solve the matrix factorization
        problem or the latent factor model to reduce users and items in a recommendation
        system to a smaller vectors in the latent space.

        Parameters
        ----------
        ratings: scipy.sparse.csr_matrix
            contains the items of the user ratings matrix in a compressed row format
        factors: int
            The number of latent factors to convert to
        regularization: float
            The regularization term
        """

        self.factors = factors
        self.regularization = regularization
        pass

    def fit(self, iterations: int=100):
        """
        Fits the ALS model to the given data

        Parameters
        ----------

        iterations: int
            the number of iterations for which to run the optimization for
        """
        pass

    def recommend_top_k_items(self, k: int, user_index: int):
        """
        Recommends the top k items for the given user_index

        Parameters
        ----------

        k: int
            the number of top items to show
        user_index: int
            the user index in the sparse ratings array
        """
        pass

    def loss_rmse(self):
        """
        Calculates and/or returns the overall loss of the model
        """
        pass

