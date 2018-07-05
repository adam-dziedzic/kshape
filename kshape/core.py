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

    >>> zscore([1, 2, 3])
    array([-1.22474487,  0.        ,  1.22474487])
    >>> zscore([1, 2, 3], ddof=1)
    array([-1.,  0.,  1.])
    >>> zscore([[1,2,3],[4,5,6]], axis=1)
    array([[-1.22474487,  0.        ,  1.22474487],
           [-1.22474487,  0.        ,  1.22474487]])
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
    :return:

    >>> roll_zeropad([1, 2, 3], shift=2)
    array([0, 0, 1])
    >>> roll_zeropad([1, 2, 3], 0)
    array([1, 2, 3])
    >>> roll_zeropad([1, 2, 3], -1)
    array([2, 3, 0])
    >>> roll_zeropad([1, 2, 3], 3)
    array([0, 0, 0])
    >>> roll_zeropad([1, 2, 3], 4)
    array([0, 0, 0])
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
    >>> _ncc_c([1,2,3,4], [1,2,3,4])
    array([ 0.13333333,  0.36666667,  0.66666667,  1.        ,  0.66666667,
            0.36666667,  0.13333333])
    >>> _ncc_c([1,1,1], [1,1,1])
    array([ 0.33333333,  0.66666667,  1.        ,  0.66666667,  0.33333333])
    >>> _ncc_c([1,2,3], [-1,-1,-1])
    array([-0.15430335, -0.46291005, -0.9258201 , -0.77151675, -0.46291005])
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
    >>> expected2 = np.array([[0.10910895, 0.43643578, 0.87287156, 0.87287156, 0.32732684], [0.35634832, 0.80178373, 0.71269665, 0.4454354 , 0.17817416], [ 0.40032038  0.80064077  0.32025631 -0.16012815 -0.08006408], [-0.10910895 -0.43643578 -0.87287156 -0.87287156 -0.32732684]])
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
    >>> _ncc_c_3dim(np.array([[1,2,3]]), np.array([[-1,-1,-1]]))
    array([[[-0.15430335, -0.46291005, -0.9258201 , -0.77151675,
             -0.46291005]]])
    >>> big = _ncc_c_3dim(np.array([[1,2,3], [-1,-1,-1], [-1,-2,-1], [1,1,1], [-1,0,-1]]), np.array([[1,2,1], [0,1,0]]))
    >>> #big
    >>> #array([[[1.09108945e-01,4.36435780e-01,8.72871561e-01,8.72871561e-01,3.27326835e-01],[-2.35702260e-01,-7.07106781e-01,-9.42809042e-01,-7.07106781e-01,-2.35702260e-01],[-1.66666667e-01,-6.66666667e-01,-1.00000000e+00,-6.66666667e-01,-1.66666667e-01],[2.35702260e-01,7.07106781e-01,9.42809042e-01,7.07106781e-01,2.35702260e-01],[-2.88675135e-01,-5.77350269e-01,-5.77350269e-01,-5.77350269e-01,-2.88675135e-01]],[[1.48359792e-17,2.67261242e-01,5.34522484e-01,8.01783726e-01,-1.48359792e-17],[-8.01234453e-18,-5.77350269e-01,-5.77350269e-01,-5.77350269e-01,8.01234453e-18],[5.66558315e-18,-4.08248290e-01,-8.16496581e-01,-4.08248290e-01,-5.66558315e-18],[8.01234453e-18,5.77350269e-01,5.77350269e-01,5.77350269e-01,-8.01234453e-18],[0.00000000e+00,-7.07106781e-01,0.00000000e+00,-7.07106781e-01,0.00000000e+00]]])
    >>> #expected = np.array([[[1.09108945e-01,4.36435780e-01,8.72871561e-01,8.72871561e-01,3.27326835e-01],[-2.35702260e-01,-7.07106781e-01,-9.42809042e-01,-7.07106781e-01,-2.35702260e-01],[-1.66666667e-01,-6.66666667e-01,-1.00000000e+00,-6.66666667e-01,-1.66666667e-01],[2.35702260e-01,7.07106781e-01,9.42809042e-01,7.07106781e-01,2.35702260e-01],[-2.88675135e-01,-5.77350269e-01,-5.77350269e-01,-5.77350269e-01,-2.88675135e-01]],[[1.48359792e-17,2.67261242e-01,5.34522484e-01,8.01783726e-01,-1.48359792e-17],[-8.01234453e-18,-5.77350269e-01,-5.77350269e-01,-5.77350269e-01,8.01234453e-18],[5.66558315e-18,-4.08248290e-01,-8.16496581e-01,-4.08248290e-01,-5.66558315e-18],[8.01234453e-18,5.77350269e-01,5.77350269e-01,5.77350269e-01,-8.01234453e-18],[0.00000000e+00,-7.07106781e-01,0.00000000e+00,-7.07106781e-01,0.00000000e+00]]])
    >>> #np.array_equal(big, expected)
    >>> big
    array([[[ 1.09108945e-01,  4.36435780e-01,  8.72871561e-01,
              8.72871561e-01,  3.27326835e-01],
            [-2.35702260e-01, -7.07106781e-01, -9.42809042e-01,
             -7.07106781e-01, -2.35702260e-01],
            [-1.66666667e-01, -6.66666667e-01, -1.00000000e+00,
             -6.66666667e-01, -1.66666667e-01],
            [ 2.35702260e-01,  7.07106781e-01,  9.42809042e-01,
              7.07106781e-01,  2.35702260e-01],
            [-2.88675135e-01, -5.77350269e-01, -5.77350269e-01,
             -5.77350269e-01, -2.88675135e-01]],
    <BLANKLINE>
           [[ 1.48359792e-17,  2.67261242e-01,  5.34522484e-01,
              8.01783726e-01, -1.48359792e-17],
            [-8.01234453e-18, -5.77350269e-01, -5.77350269e-01,
             -5.77350269e-01,  8.01234453e-18],
            [ 5.66558315e-18, -4.08248290e-01, -8.16496581e-01,
             -4.08248290e-01, -5.66558315e-18],
            [ 8.01234453e-18,  5.77350269e-01,  5.77350269e-01,
              5.77350269e-01, -8.01234453e-18],
            [ 0.00000000e+00, -7.07106781e-01,  0.00000000e+00,
             -7.07106781e-01,  0.00000000e+00]]])
    >>> x = np.array([[1,2,3],[4,5,6],[1,0,2]])
    >>> centroids = np.array([[1,1,1],[1,0,2]])
    >>> _ncc_c_3dim(x, centroids)
    array([[[0.15430335, 0.46291005, 0.9258201 , 0.77151675, 0.46291005],
            [0.26318068, 0.59215653, 0.98692754, 0.72374686, 0.39477102],
            [0.25819889, 0.25819889, 0.77459667, 0.51639778, 0.51639778]],
    <BLANKLINE>
           [[0.23904572, 0.47809144, 0.83666003, 0.23904572, 0.35856858],
            [0.40771775, 0.50964719, 0.81543551, 0.2548236 , 0.30578831],
            [0.4       , 0.        , 1.        , 0.        , 0.4       ]]])
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
    >>> _sbd([1,1,1], [1,1,1])
    (-2.2204460492503131e-16, array([1, 1, 1]))
    >>> _sbd([0,1,2], [1,2,3])
    (0.043817112532485103, array([1, 2, 3]))
    >>> _sbd([1,2,3], [0,1,2])
    (0.043817112532485103, array([0, 1, 2]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax()
    dist = 1 - ncc[idx]
    yshift = roll_zeropad(y, (idx + 1) - max(len(x), len(y)))

    return dist, yshift


def _extract_shape(idx, x, j, cur_center):
    """
    >>> _extract_shape(np.array([0,1]), np.array([[1,2,3], [4,5,6]]), 1, np.array([0,3,4]))
    array([-1.,  0.,  1.])
    >>> _extract_shape(np.array([0,1]), np.array([[-1,2,3], [4,-5,6]]), 1, np.array([0,3,4]))
    array([-0.96836405,  1.02888681, -0.06052275])
    >>> _extract_shape(np.array([1,0,1,0]), np.array([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]]), 0, np.array([0,0,0,0]))
    array([-1.2089303 , -0.19618238,  0.19618238,  1.2089303 ])
    >>> _extract_shape(np.array([0,0,1,0]), np.array([[1,2,3,4],[0,1,2,3],[-1,1,-1,1],[1,2,2,3]]), 0, np.array([-1.2089303,-0.19618238,0.19618238,1.2089303]))
    array([-1.19623139, -0.26273649,  0.26273649,  1.19623139])
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
        # # tensor.max in PyTorch returns a tuple. The first return element in the tuple is the maximum value of each
        # # row of the input tensor in the given dimension dim. The second return value is the index location of each
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
    :return: centroids, and for each centroid the index of the member time-series (from the input)

    >>> from numpy.random import seed; seed(31)
    >>> a = [[0,1,2,3,4], [1,2,3,4,5],[2,3,4,5,6], [3,4,5,6,7],[4,5,6,7,8]]
    >>> results = kshape(a, 2)
    >>> # check the results: we expect 2 clusters
    >>> # expected: [(array([-1.05204252, -1.05204252,  0.30722438,  0.70136168,  1.09549899]), [3, 4]), (array([-1.26491106e+00, -6.32455532e-01,  7.47745081e-17,  6.32455532e-01, 1.26491106e+00]), [0, 1, 2])]
    >>> first_cluster = results[0]
    >>> second_cluster = results[1]
    >>> first_cluster[0]
    >>> np.testing.assert_array_almost_equal(first_cluster[0], np.array([-1.05204252, -1.05204252,  0.30722438,  0.70136168,  1.09549899]))
    >>> np.testing.assert_array_equal(np.array(first_cluster[1]), np.array([3, 4]))
    >>> np.testing.assert_array_almost_equal(np.array(second_cluster[0]), np.array([-1.26491106e+00, -6.32455532e-01,  7.47745081e-17,  6.32455532e-01, 1.26491106e+00]))
    >>> np.testing.assert_array_equal(np.array(second_cluster[1]), np.array([0, 1, 2]))
    """
    idx, centroids = _kshape(np.array(x), k)
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
