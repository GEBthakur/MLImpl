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
from tqdm import tqdm


def loss_als(ratings, rows, columns, users, items, regular):
    """
    Computes the loss of the current ratings approximation R ~ UV^T

    Parameters
    ----------
    ratings: scipy.sparse.csr_matrix
        the sparse user-item explicit ratings matrix
    rows: np.array
        array containing indices of the rows of the non-zero elements in ratings.
        This is required for the loss function evaluations.
    rows: np.array
        array containing indices of the columns of the non-zero elements in ratings
    users: np.ndarray
        the latent user matrix
    items: np.ndarray
        the latent item matrix
    regular: float
        the regularization coefficient
    
    Returns
    -------
    loss: float
        the regularized ALS loss using approximation of the ratings matrix
    """

    actual_ratings = ratings.data

    # Compute predictions for all non-zero entries
    predictions = np.sum(users[rows] * items[columns], axis=1)

    # Calculate the squared differences
    squared_diff = (actual_ratings - predictions)**2

    # frobenius norm of user and item latent matrices
    fro_user = np.linalg.norm(users, ord='fro')**2
    fro_item = np.linalg.norm(items, ord='fro')**2

    # Compute the loss
    loss = 0.5*(np.sum(squared_diff) + regular*(fro_user + fro_item))
    return loss


def A_inv(mat, regular):
    """
    Computes the inverse of SUM(UU^T) + Lambda I to be used in each inner iteration of ALS
    uu^T = [u_1, u_2, u_3][u_1 
                        u_2
                        u_3]
    regular: float
        the regularization coefficient
    Parameters
    ----------
    mat: np.ndarray
        the user or item latent matrix depending on the sub iteration
    """
    uut = np.einsum('ij,ik->jk', mat, mat)
    return np.linalg.inv(uut + regular*np.eye(uut.shape[0]))


class ALSNumExpl:
    def __init__(self, ratings: csr_matrix, factors: int, regularization: float, seed: int=42):
        """Alternating Least Squares - NumPy Variant - Explicit ratings only

        Implements the alternating least squares algorithm to solve the matrix factorization
        problem or the latent factor model to reduce users and items in a recommendation
        system to a smaller vectors in the latent space. This isn't a very robust implementation
        that handles missing values in ratings matrix. Assumes all values of ratings are correct.

        This is a personal implementation to model alternative least squares on small datasets.
        Focus is more on implementation from scratch of the models rather than providing an all
        purpose interface to ALS. There will also be a tensorflow based model for this. Comparison
        shall be done on the overall runtimes of these models.

        Parameters
        ----------
        ratings: scipy.sparse.csr_matrix
            contains the items of the user ratings matrix in a compressed row format
        factors: int
            The number of latent factors to convert to
        regularization: float
            The regularization term
        seed: float
            the random seed for reproducibility
        """

        self.ratings = ratings
        self.factors = factors
        self.regularization = regularization
        self.seed = seed

        # check if the factors is not more than the size of the user-rating matrix
        if (factors > ratings.shape[0]) or (factors > ratings.shape[1]):
            raise ValueError("The number of factors cannot be larger than unique count of users or items")
        
        self.user_count = ratings.shape[0]
        self.item_count = ratings.shape[1]

        # regularization term can't be negative
        if regularization < 0:
            raise ValueError("Regularization factor can't be negative")

        # initializting the user and item latent factor matrices
        # using a uniform random small number. Gaussian centered around zero can also be used.
        # TODO: add Gaussian initial parameters.

        np.random.seed(seed)

        # TODO: In Large-Scale Parallel Collaborative Filtering for the Netflix Prize Y Zhou(2008),
        # the authors suggest keeping the average rating of each item in ilf's first components
        self.ulf = np.random.rand(self.user_count, factors)*0.001

        ilf = np.random.rand(self.item_count, self.factors - 1)*0.001
        item_avg = ratings.mean(axis=0)[0].A[0].reshape(-1,1)

        self.ilf = np.hstack((item_avg, ilf))

        # array to store losses
        self.loss = []


    def fit(self, iterations: int=10):
        """
        Fits the ALS model to the given data

        Parameters
        ----------

        iterations: int
            the number of iterations for which to run the optimization for
        """
        rows, columns = self.ratings.nonzero()

        # initial loss
        initial_loss = loss_als(self.ratings,
                                rows,
                                columns,
                                self.ulf,
                                self.ilf,
                                self.regularization)
        self.loss.append(initial_loss)

        for _ in tqdm(range(iterations)):
            # first update user matrix
            for i in range(self.user_count):
                # find items rated by user i
                Iui = self.ilf[columns[rows == i]]
                a_inv = A_inv(Iui, self.regularization)
                B = self.ratings.getrow(i).data.dot(Iui)
                self.ulf[i] = B @ a_inv.T

            # update item matrix
            for j in range(self.item_count):
                # find users who rated by item j
                uIj = self.ulf[rows[columns == j]]
                a_inv = A_inv(uIj, self.regularization)
                B = self.ratings.getcol(j).data.dot(uIj)
                self.ilf[j] = B @ a_inv.T

            iteration_loss = loss_als(self.ratings,
                                      rows,
                                      columns,
                                      self.ulf,
                                      self.ilf,
                                      self.regularization)
            self.loss.append(iteration_loss)
        return self.loss


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


