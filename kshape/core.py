import math
import numpy as np

from numpy.random import randint
from numpy.linalg import norm, eigh
from numpy.fft import fft, ifft


def zscore(a, axis=0, ddof=0):
    """
    Z-normalize signal a.

    :param a: the input signal (time-series)
    :param axis: axis for the normalization
    :param ddof: Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the
    number of elements. By default ddof is zero.
    :return: z-normalized a

    >>> result = zscore(np.array([1., 2., 3.]))
    >>> np.testing.assert_array_almost_equal(result, np.array([-1.2247,  0.0000,  1.2247]), decimal=4)
    >>> result = zscore(np.array([1., 2., 3.]), ddof=1)
    >>> np.testing.assert_array_almost_equal(result, np.array([-1.,  0.,  1.]), decimal=4)
    >>> result = zscore(np.array([[1.,2.,3.],[4.,5.,6.]]), axis=1)
    >>> np.testing.assert_array_almost_equal(result, np.array([[-1.2247,  0.0000,  1.2247], [-1.2247,  0.0000,  1.2247]]), decimal=4)
    """
    a = np.asanyarray(a)
    mns = a.mean(axis=axis)
    sstd = a.std(axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd
    return np.nan_to_num(res)


def roll_zeropad(a, shift, axis=None):
    """
    Shift of a sequence with zero padding.

    :param a: input time-series (sequence)
    :param shift: the number of positions to be shifted to the right (shift >= 0) or to the left (shift < 0)
    :param axis: the axis
    :return: shifted time-series a

    >>> result = roll_zeropad(np.array([1., 2., 3.]), shift=2)
    >>> np.testing.assert_array_equal(result, np.array([0., 0., 1.]))
    >>> result=roll_zeropad(np.array([1, 2, 3]), 0)
    >>> np.testing.assert_array_equal(result, np.array([1, 2, 3]))
    >>> result=roll_zeropad(np.array([1., 2., 3.]), -1)
    >>> np.testing.assert_array_equal(result, np.array([2., 3., 0.]))
    >>> result=roll_zeropad(np.array([1., 2., 3.]), 3)
    >>> np.testing.assert_array_equal(result, np.array([0., 0., 0.]))
    >>> result=roll_zeropad(np.array([1, 2, 3]), 4)
    >>> np.testing.assert_array_equal(result, np.array([0, 0, 0]))
    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def _ncc_c(x, y):
    """
    Variant of NCCc that operates with 1 dimensional x array and 1 dimensional y array.

    :param x: one-dimensional array
    :param y: one-dimensional array
    :return: normalized cross correlation (with coefficient normalization)

    >>> result1 = _ncc_c(np.array([1.,2.,3.,4.]), np.array([1.,2.,3.,4.]))
    >>> expected1 = np.array([0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333])
    >>> np.testing.assert_array_almost_equal(result1, expected1, decimal=4)
    >>> result2 = _ncc_c(np.array([1.,1.,1.]), np.array([1.,1.,1.]))
    >>> expected2 = np.array([0.3333, 0.6667, 1.0000, 0.6667, 0.3333])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c(np.array([1.,2.,3.]), np.array([-1.,-1.,-1.]))
    >>> expected3 = np.array([-0.1543, -0.4629, -0.9258, -0.7715, -0.4629])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    >>> result4 = _ncc_c(np.array([1.1,5.5,3.9,1.0]), np.array([9.1,-1.1,0.7,-1.3]))
    >>> expected4 = np.array([-0.0223, -0.0995, -0.0379, 0.0841, 0.7248, 0.5365, 0.142])
    >>> np.testing.assert_array_almost_equal(result4, expected4, decimal=4)
    """
    den = np.array(norm(x) * norm(y))
    den[den == 0] = np.Inf

    x_len = len(x)
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]))
    return np.real(cc) / den


def _ncc_c_2dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 1 dimensional
    y vector.

    :param x: 2 dimensional array with time series
    :param y: 1 dimensional array with a single centroid
    :return: a 2 dimensional array of normalized fourier transforms

    >>> result1 = _ncc_c_2dim(np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]]), np.array([1.0, 2.0, 1.0]))
    >>> expected1 = np.array([[0.10910895, 0.43643578, 0.87287156, 0.87287156, 0.32732684], [0.35634832, 0.80178373, 0.71269665, 0.4454354 , 0.17817416]])
    >>> np.testing.assert_almost_equal(result1, expected1)

    >>> result2 = _ncc_c_2dim(np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0], [5.0, 0.0, -1.0], [-1.0, -2.0, -3.0]]), np.array([1.0, 2.0, 1.0]))
    >>> expected2 = np.array([[0.10910895, 0.43643578, 0.87287156, 0.87287156, 0.32732684], [0.35634832, 0.80178373, 0.71269665, 0.4454354 , 0.17817416], [ 0.40032038,  0.80064077,  0.32025631, -0.16012815, -0.08006408], [-0.10910895, -0.43643578, -0.87287156, -0.87287156, -0.32732684]])
    >>> np.testing.assert_almost_equal(result2, expected2)
    """
    den = np.array(norm(x, axis=1) * norm(y))
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size)))
    cc = np.concatenate((cc[:,-(x_len-1):], cc[:,:x_len]), axis=1)
    return np.real(cc) / den[:, np.newaxis]


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 2 dimensional
    y vector

    Returns a 3 dimensional array of normalized fourier transforms

    >>> result = _ncc_c_3dim(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 0.0, 2.0]]), np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 2.0]]))
    >>> expected = np.array([[[0.1543, 0.4629, 0.9258, 0.7715, 0.4629], [0.2632, 0.5922, 0.9869, 0.7237, 0.3948], [0.2582, 0.2582, 0.7746, 0.5164, 0.5164]], [[0.2390, 0.4781, 0.8367, 0.2390, 0.3586], [0.4077, 0.5096, 0.8154, 0.2548, 0.3058], [0.4000, 0.0000, 1.0000, 0.0000, 0.4000]]])
    >>> np.testing.assert_array_almost_equal(result, expected, decimal=4)
    >>> result2 = _ncc_c_3dim(np.array([[1.,2.,3.]]), np.array([[-1.,-1.,-1.]]))
    >>> expected2 = np.array([[[-0.1543, -0.4629, -0.9258, -0.7715, -0.4629]]])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c_3dim(np.array([[1.,2.,3.,4.]]), np.array([[1.,2.,3.,4.]]))
    >>> expected3 = np.array([[[0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333]]])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    >>> result4 = _ncc_c_3dim(np.array([[1.,1.,1.]]), np.array([[1.,1.,1.]]))
    >>> expected4 = np.array([[[0.3333, 0.6667, 1.0000, 0.6667, 0.3333]]])
    >>> np.testing.assert_array_almost_equal(result4, expected4, decimal=4)
    """
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2*x_len-1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:,:,-(x_len-1):], cc[:,:,:x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd(x, y):
    """
    Shape based distance between x and y.

    :param x: the first time-series
    :param y: the seconda time-series
    :return: shape based distance between x and y, shifted y to the position that maximizes the similarity of x and y

    >>> dist, y = _sbd(np.array([1.,1.,1.]), np.array([1.,1.,1.]))
    >>> np.testing.assert_array_almost_equal(dist, np.array(0.), decimal=4)
    >>> np.testing.assert_array_equal(y, np.array([1., 1., 1.]))
    >>> dist, y = _sbd(np.array([0.,1.,2.]), np.array([1.,2.,3.]))
    >>> np.testing.assert_array_almost_equal(dist, np.array(0.0438), decimal=4)
    >>> np.testing.assert_array_equal(y, np.array([1., 2., 3.]))
    >>> dist, y = _sbd(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
    >>> np.testing.assert_array_almost_equal(dist, np.array(0.0438), decimal=4)
    >>> np.testing.assert_array_equal(y, np.array([0., 1., 2.]))
    >>> dist, y = _sbd(np.array([1.,2.,3.]), np.array([-1.,-1.,-1.]))
    >>> np.testing.assert_array_almost_equal(dist, np.array(1.1543), decimal=4)
    >>> np.testing.assert_array_equal(y, np.array([-1.,  0.,  0.]))
    >>> dist, y = _sbd(np.array([1.,2.,3.]), np.array([0.,1.,2.]))
    >>> np.testing.assert_array_almost_equal(dist, np.array(0.0438), decimal=4)
    >>> np.testing.assert_array_equal(y, np.array([0., 1., 2.]))
    >>> dist, y = _sbd(np.array([0., 0., 1., 2., 3., 0., 0.]), np.array([1., 2., 3., 0., 0., 0., 0.]))
    >>> np.testing.assert_array_almost_equal(dist, np.array(5.9605e-08), decimal=4)
    >>> np.testing.assert_array_equal(y, np.array([0., 0., 1., 2., 3., 0., 0.]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


def _extract_shape(idx, x, j, cur_center):
    """
    Find new centroid.

    :param idx: array of cluster numbers for each time-series
    :param x: the input time-series
    :param j: the current clustered to be considered
    :param cur_center: the current centroid for the cluter j
    :return: new centroid for cluster j

    >>> result = _extract_shape(np.array([0, 1]), np.array([[1., 2., 3.], [4., 5., 6.]]), 2, np.array([0., 3., 4.]))
    >>> np.testing.assert_array_almost_equal(result, np.array([[0., 0., 0.]]), decimal=4)
    >>> result = _extract_shape(np.array([0, 1]), np.array([[1., 2., 3.], [4., 5., 6.]]), 1, np.array([0., 3., 4.]))
    >>> np.testing.assert_array_almost_equal(result, np.array([-1.,  0.,  1.]), decimal=4)
    >>> result = _extract_shape(np.array([0, 1]), np.array([[-1., 2., 3.], [4., -5., 6.]]), 1, np.array([0., 3., 4.]))
    >>> np.testing.assert_array_almost_equal(result, np.array([-0.9684,  1.0289, -0.0605]), decimal=4)
    >>> result = _extract_shape(np.array([1, 0, 1, 0]), np.array([[1., 2., 3., 4.], [0., 1., 2., 3.], [-1., 1., -1., 1.], [1., 2., 2., 3.]]), 0, np.array([0., 0., 0., 0.]))
    >>> np.testing.assert_array_almost_equal(result, np.array([-1.2089, -0.1962,  0.1962,  1.2089]), decimal=4)
    >>> result = _extract_shape(np.array([0, 0, 1, 0]), np.array([[1., 2., 3., 4.],[0., 1., 2., 3.],[-1., 1., -1., 1.],[1., 2., 2., 3.]]), 0, np.array([-1.2089303, -0.19618238, 0.19618238, 1.2089303]))
    >>> np.testing.assert_array_almost_equal(result, np.array([-1.1962, -0.2627,  0.2627,  1.1962]), decimal=4)
    """
    _a = []
    for i in range(len(idx)):
        if idx[i] == j:
            if cur_center.sum() == 0:
                opt_x = x[i]
            else:
                _, opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    a = np.array(_a)

    if len(a) == 0:
        return np.zeros((1, x.shape[1]))
    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = np.dot(y.transpose(), y)

    p = np.empty((columns, columns))
    p.fill(1.0/columns)
    p = np.eye(columns) - p

    m = np.dot(np.dot(p, s), p)
    _, vec = eigh(m)
    centroid = vec[:, -1]
    finddistance1 = math.sqrt(((a[0] - centroid) ** 2).sum())
    finddistance2 = math.sqrt(((a[0] + centroid) ** 2).sum())

    if finddistance1 >= finddistance2:
        centroid *= -1

    return zscore(centroid, ddof=1)


def _kshape(x, k, max_iterations=100, idx=None):
    """
    The main call of kshape.

    :param x: the 2 dimensional array with time-series data
    :param k: the scalar with number of expected clusters
    :param max_iterations: how many times iterate through the time-series data, where each iteration is composed of two
    steps: 1) cluster membership assignment, 2) centroid computation
    :param idx: the initial assignment of time series to clusters
    :return: a two element tuple where at the first position we have (for each time series) - its index of a cluster
    and, in the second element of the tuple, the centroids for each cluster.

    >>> # from numpy.random import seed; seed(0)  # no need to set the seed - we set the initial cluster assignment

    >>> result_cluster_assignment, result_centroids = _kshape(np.array([[1.0,2.0,3.0,4.0], [0.0,1.0,2.0,3.0], [-1.0,1.0,-1.0,1.0], [1.0,2.0,2.0,3.0], [1.0,2.2,-2.0,-3.0], [-1.1,2.3,-2.9,3.4]]), 3, idx=np.array([1, 2, 1, 0, 0, 1]))
    >>> expected_cluster_assignments, expected_centroids = (np.array([2, 2, 1, 0, 0, 1]), np.array([[-0.663535, -1.008225,  0.565868,  1.105892], [-0.701075,  0.761482, -1.011736,  0.95133 ], [-1.161895, -0.387298,  0.387299,  1.161895]]))
    >>> np.testing.assert_array_equal(result_cluster_assignment, expected_cluster_assignments)
    >>> np.testing.assert_array_almost_equal(result_centroids, expected_centroids)

    >>> result_cluster_assignment, result_centroids = _kshape(np.array([[1.0,2.0,3.0,4.0], [0.0,1.0,2.0,3.0], [-1.0,1.0,-1.0,1.0], [1.0,2.0,2.0,3.0]]), 2, idx=np.array([1, 0, 1, 0]))
    >>> expected_cluster_assigments, expected_centroids = (np.array([0, 0, 1, 0]), np.array([[-1.050464, -0.524116,  0.350155,  1.224426], [-0.866025,  0.866025, -0.866025,  0.866025]]))
    >>> np.testing.assert_equal(result_cluster_assignment, expected_cluster_assigments)
    >>> np.testing.assert_array_almost_equal(result_centroids, expected_centroids)
    """
    m = x.shape[0]
    if idx is None:
        idx = randint(0, k, size=m)
    centroids = np.zeros((k, x.shape[1]))

    for _ in range(max_iterations):
        old_idx = idx.copy()
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])

        # distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T

        # similarities = _ncc_c_3dim(x, centroids)
        # # np.array.max in PyTorch returns a tuple. The first return element in the tuple is the maximum value of each
        # # row of the input np.array in the given dimension dim. The second return value is the index location of each
        # # maximum value found (argmax).
        # max_similarities = similarities.max(axis=2)
        # distances = (1 - max_similarities).T

        # idx = distances.argmin(1)

        for i, time_series in enumerate(x):
            similarities = _ncc_c_2dim(centroids, time_series)
            max_similarities = similarities.max(axis=-1)
            closest_centroid_index = max_similarities.argmax()
            idx[i] = closest_centroid_index

        if np.array_equal(old_idx, idx):
            break

    return idx, centroids


def kshape(x, k, max_iterations=100, idx=None):
    """
    Run kshape.

    :param x: the input time-series
    :param k: the number of expected clusters
    :param max_iterations: max number of iterations to run kshape
    :param idx: initial assignment of time-series to clusters
    :return: centroids, and for each centroid the index of the member time-series

    >>> from numpy.random import seed; seed(31)
    >>> a = [[0,1,2,3,4], [1,2,3,4,5],[5.1,3,4,5,1.0], [3,4,5,3.4,7],[4,5,6,7,8]]
    >>> results = kshape(a, 2, idx=[1, 0, 1, 1, 0])
    >>> first_cluster = results[0]
    >>> second_cluster = results[1]
    >>> np.testing.assert_array_almost_equal(first_cluster[0], np.array([-1.24937 , -0.587328,  0.074714,  0.363188,  1.398797]))
    >>> np.testing.assert_array_equal(first_cluster[1], np.array([0, 1, 3, 4]))
    >>> np.testing.assert_array_almost_equal(second_cluster[0], np.array([-0.447214, -0.447214, -0.447214, -0.447214,  1.788854]))
    >>> np.testing.assert_array_equal(second_cluster[1], np.array([2]))
    """
    idx, centroids = _kshape(np.array(x), k, max_iterations, idx)
    clusters = []
    for i, centroid in enumerate(centroids):
        series = []
        for j, val in enumerate(idx):
            if i == val:
                series.append(j)
        clusters.append((centroid, series))
    return clusters


if __name__ == "__main__":
    import sys
    import doctest
    sys.exit(doctest.testmod()[0])
