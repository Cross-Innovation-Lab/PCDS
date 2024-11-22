import numpy as np
from sklearn.cluster import KMeans


class SpectralClusterer:
    def __init__(self, min_clusters=1, max_clusters=None, max_iter=300, row_wise_renorm=False) -> None:
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.row_wise_renorm = row_wise_renorm

    def _refine(self, affinity):
        refined_affinity = np.copy(affinity)
        # np.fill_diagonal(refined_affinity, 0.0)
        di = np.diag_indices(refined_affinity.shape[0])
        refined_affinity[di] = refined_affinity.max(axis=1)
        refined_affinity = np.maximum(refined_affinity, refined_affinity.T)
        # refined_affinity = np.matmul(refined_affinity, refined_affinity.T)
        row_max = refined_affinity.max(axis=1)
        refined_affinity /= np.expand_dims(row_max, axis=1)
        return refined_affinity

    def _compute_laplacian(self, affinity):
        degree = np.diag(np.sum(affinity, axis=1))
        laplacian = degree - affinity
        return laplacian

    def _compute_sorted_eigenvectors(self, input_matrix, descend=True):
        """Sort eigenvectors by the real part of eigenvalues.
        Args:
            input_matrix: the matrix to perform eigen analysis with shape (M, M)
            descend: sort eigenvalues in a descending order. Default is True
        Returns:
            w: sorted eigenvalues of shape (M,)
            v: sorted eigenvectors, where v[;, i] corresponds to ith largest
            eigenvalue
        """
        # Eigen decomposition.
        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)
        eigenvalues = eigenvalues.real
        eigenvectors = eigenvectors.real
        if descend:
            # Sort from largest to smallest.
            index_array = np.argsort(-eigenvalues)
        else:
            # Sort from smallest to largest.
            index_array = np.argsort(eigenvalues)
        # Re-order.
        w = eigenvalues[index_array]
        v = eigenvectors[:, index_array]
        return w, v

    def _compute_number_of_clusters(self,
                                    eigenvalues,
                                    max_clusters=None,
                                    stop_eigenvalue=1e-2,
                                    descend=True,
                                    eps=1e-10):
        """Compute number of clusters using EigenGap principle.
        Use maximum EigenGap principle to find the number of clusters.
        Args:
            eigenvalues: sorted eigenvalues of the affinity matrix
            max_clusters: max number of clusters allowed
            stop_eigenvalue: we do not look at eigen values smaller than this
            eigengap_type: the type of the eigengap computation
            descend: sort eigenvalues in a descending order. Default is True
            eps: a small value for numerial stability
        Returns:
            max_delta_index: number of clusters as an integer
            max_delta_norm: normalized maximum eigen gap
        """
        max_delta = 0
        max_delta_index = 0
        range_end = len(eigenvalues)
        if max_clusters and max_clusters + 1 < range_end:
            range_end = max_clusters + 1

        if not descend:
            # The first eigen value is always 0 in an ascending order
            for i in range(1, range_end - 1):
                delta = eigenvalues[i + 1] / (eigenvalues[i] + eps)
                if delta > max_delta:
                    max_delta = delta
                    max_delta_index = i + 1  # Index i means i+1 clusters
        else:
            for i in range(1, range_end):
                if eigenvalues[i - 1] < stop_eigenvalue:
                    break
                delta = eigenvalues[i - 1] / (eigenvalues[i] + eps)
                if delta > max_delta:
                    max_delta = delta
                    max_delta_index = i

        return max_delta_index, max_delta

    def _run_kmeans(self, spectral_embeddings, n_clusters, max_iter):
        kmeans_clusterer = KMeans(n_clusters=n_clusters, init="k-means++",n_init='auto', max_iter=max_iter, random_state=0)
        labels = kmeans_clusterer.fit_predict(spectral_embeddings)
        return labels

    def predict(self, affinity):
        '''
        affinity: seq_num, seq_num
        '''
        seq_num = affinity.shape[0]
        # print(f'seqnum:{seq_num}')
        if seq_num == 1:
            return np.array([0]), 1
        # print(f'affinity:{affinity}')
        affinity = self._refine(affinity)
        # print(f'refinred_affinity:{affinity}')
        laplacian = self._compute_laplacian(affinity)
        # print(f'laplacian:{laplacian}')
        eigenvalues, eigenvectors = self._compute_sorted_eigenvectors(laplacian, descend=False)
        n_clusters, max_delta_norm = self._compute_number_of_clusters(eigenvalues,
                                                                      max_clusters=self.max_clusters,
                                                                      descend=False)
        if self.min_clusters is not None:
            n_clusters = max(n_clusters, self.min_clusters)
        spectral_embeddings = eigenvectors[:, :n_clusters]  # [seq_num, n_clusters]
        if self.row_wise_renorm:
            rows_norm = np.linalg.norm(spectral_embeddings, axis=1, ord=2)
            spectral_embeddings = spectral_embeddings / np.reshape(rows_norm, (seq_num, 1))
        labels = self._run_kmeans(spectral_embeddings, n_clusters, max_iter=self.max_iter)
        num_unique = len(np.unique(labels))
        return labels, num_unique
