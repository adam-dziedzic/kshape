import numpy as np
import torch
from torch import tensor, norm, rfft, irfft, cat, mul, add, div, stack, symeig, sqrt

"""
Implementation of kShape using PyTorch: https://github.com/pytorch/pytorch for GPU support.
"""


def _torch_version():
    """
    Torch version.

    :return: the major number of the torch version
    """
    version = None
    if "__version__" in dir(torch):
        version = torch.__version__
    if version is not None:
        try:
            version = float(".".join(torch.__version__.split(".")[:2]))
        except ValueError:
            version = None
    return version


def flip1(x, dim=0):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension according to which we flip the tensor
    :return: flipped tensor
    >>> flip2(tensor([1, 2, 3]), dim=0)
    tensor([3, 2, 1])
    """
    return x.flip([dim])


def flip2(x, dim=0):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension according to which we flip the tensor
    :return: flipped tensor

    This flip method is used only for version of PyTorch <= 0.4. There is a flip method added to Tensor in PyTorch 5.0.

    >>> flip2(tensor([1, 2, 3]), dim=0)
    tensor([3, 2, 1])
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


flip = flip1

_torch_version = _torch_version()
if _torch_version is None or _torch_version <= 0.4:
    flip = flip2


def zscore(a, axis=0, ddof=0):
    """
    Z-normalize signal a.

    :param a: the input signal (time-series)
    :param axis: axis for the normalization
    :param ddof: Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the
    number of elements. By default ddof is zero.
    :return: z-normalized a

    >>> zscore(tensor([1., 2., 3.]))
    tensor([-1.2247,  0.0000,  1.2247])
    >>> zscore(tensor([1., 2., 3.]), ddof=1)
    tensor([-1.,  0.,  1.])
    >>> zscore(tensor([[1.,2.,3.],[4.,5.,6.]]), axis=1)
    tensor([[-1.2247,  0.0000,  1.2247],
            [-1.2247,  0.0000,  1.2247]])
    """
    mns = a.mean(dim=axis)
    sstd = a.std(dim=axis, unbiased=(ddof == 1))
    if axis and mns.dim() < a.dim():
        return (a - mns.unsqueeze(axis).expand(a.size())).div(sstd.unsqueeze(axis).expand(a.size()))
    else:
        return a.sub_(mns).div(sstd)


def roll_zeropad(a, shift):
    """
    Shift of a sequence with zero padding.

    :param a: input time-series (sequence)
    :param shift: the number of positions to be shifted to the right (shift >= 0) or to the left (shift < 0)
    :return:

    >>> roll_zeropad(tensor([1., 2., 3.]), shift=2)
    tensor([0., 0., 1.])
    >>> roll_zeropad(tensor([1, 2, 3]), 0)
    tensor([1, 2, 3])
    >>> roll_zeropad(tensor([1., 2., 3.]), -1)
    tensor([2., 3., 0.])
    >>> roll_zeropad(tensor([1., 2., 3.]), 3)
    tensor([0., 0., 0.])
    >>> roll_zeropad(tensor([1, 2, 3]), 4)
    tensor([0, 0, 0])
    """
    if shift == 0:
        return a
    if abs(shift) > len(a):
        return torch.zeros_like(a)
    padding = torch.zeros(abs(shift), device=a.device, dtype=a.dtype)
    if shift < 0:
        return torch.cat((a[abs(shift):], padding))
    else:
        return torch.cat((padding, a[:-shift]))


def complex_mul(x, y):
    """
    Multiply arrays of complex numbers. Each complex number is expressed as a pair of real and imaginary parts.

    :param x: the first array of complex numbers
    :param y: the second array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural Networks (https://arxiv.org/pdf/1509.09308.pdf)
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> complex_mul(x, y)
    tensor([[108.,   0.],
            [ -8.,  16.],
            [ 12.,   0.]])
    """
    ua = x[:, 0]
    va = y[:, 0]
    ub = x[:, 0] + x[:, 1]
    vb = y[:, 1]
    uc = x[:, 1] - x[:, 0]
    vc = y[:, 0] + y[:, 1]
    result = torch.empty_like(x)
    uavc = mul(ua, vc)
    result[:, 0] = add(uavc, mul(mul(ub, vb), -1))
    result[:, 1] = add(mul(uc, va), uavc)
    return result


def _ncc_c(x, y):
    """
    # Variant of NCCc that operates with 1 dimensional x array and 1 dimensional y array.
    :param x: one-dimensional array
    :param y: one-dimensional array
    :return: normalized cross correlation (with coefficient normalization)

    >>> _ncc_c(tensor([1.,2.,3.,4.]), tensor([1.,2.,3.,4.]))
    tensor([0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333])
    >>> _ncc_c(tensor([1.,1.,1.]), tensor([1.,1.,1.]))
    tensor([0.3333, 0.6667, 1.0000, 0.6667, 0.3333])
    >>> _ncc_c(tensor([1.,2.,3.]), tensor([-1.,-1.,-1.]))
    tensor([-0.1543, -0.4629, -0.9258, -0.7715, -0.4629])
    """
    # the denominator for normalization
    den = norm(x) * norm(y)
    if den == 0:
        den = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    signal_ndim = 1
    x_len = len(x)
    fft_size = 1 << (2 * x_len - 1).bit_length()
    pad_size = fft_size - x_len
    padding = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
    # conjugate in the frequency domain is equivalent to the reversed signal in the time domain
    y = flip(y, dim=0)
    x = cat((x, padding))
    y = cat((y, padding))
    xfft = rfft(x, signal_ndim)
    yfft = rfft(y, signal_ndim)

    cc = irfft(complex_mul(xfft, yfft), signal_ndim=signal_ndim, signal_sizes=(fft_size,))
    return div(cc[:2 * x_len - 1], den)


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional x arrays and 2 dimensional y arrays.
    Returns a 3 dimensional array of normalized fourier transforms.


    """
    den = norm(x, axis=1)[:, None] * norm(y, axis=1)
    den[den == 0] = np.Inf
    x_len = x.shape[-1]
    fft_size = 1 << (2 * x_len - 1).bit_length()
    cc = ifft(fft(x, fft_size) * np.conj(fft(y, fft_size))[:, None])
    cc = np.concatenate((cc[:, :, -(x_len - 1):], cc[:, :, :x_len]), axis=2)
    return np.real(cc) / den.T[:, :, None]


def _sbd(x, y):
    """
    Shape based distance between x and y.

    :param x: the first time-series
    :param y: the seconda time-series
    :return: shape based distance between x and y, shifted y to the position that maximizes the similarity of x and y

    >>> _sbd(tensor([1.,1.,1.]), tensor([1.,1.,1.]))
    (tensor(0.), tensor([1., 1., 1.]))
    >>> _sbd(tensor([0.,1.,2.]), tensor([1.,2.,3.]))
    (tensor(0.0438), tensor([1., 2., 3.]))
    >>> _sbd(tensor([1.,2.,3.]), tensor([0.,1.,2.]))
    (tensor(0.0438), tensor([0., 1., 2.]))
    >>> _sbd(tensor([1.,2.,3.]), tensor([-1.,-1.,-1.]))
    (tensor(1.1543), tensor([-1.,  0.,  0.]))
    >>> _sbd(tensor([1.,2.,3.], dtype=torch.float64), tensor([0.,1.,2.], dtype=torch.float64))
    (tensor(0.0438, dtype=torch.float64), tensor([0., 1., 2.], dtype=torch.float64))
    >>> _sbd(tensor([0., 0., 1., 2., 3., 0., 0.]), tensor([1., 2., 3., 0., 0., 0., 0.]))
    (tensor(5.9605e-08), tensor([0., 0., 1., 2., 3., 0., 0.]))
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

    >>> _extract_shape(tensor([0, 1]), tensor([[1., 2., 3.], [4., 5., 6.]]), 2, tensor([0., 3., 4.]))
    tensor([0., 0., 0.])
    >>> _extract_shape(tensor([0, 1]), tensor([[1., 2., 3.], [4., 5., 6.]]), 1, tensor([0., 3., 4.]))
    tensor([-1.,  0.,  1.])
    >>> _extract_shape(tensor([0, 1]), tensor([[-1., 2., 3.], [4., -5., 6.]]), 1, tensor([0., 3., 4.]))
    tensor([-0.9684,  1.0289, -0.0605])
    >>> _extract_shape(tensor([1, 0, 1, 0]), tensor([[1., 2., 3., 4.], [0., 1., 2., 3.], [-1., 1., -1., 1.], [1., 2., 2., 3.]]), 0, tensor([0., 0., 0., 0.]))
    tensor([-1.2089, -0.1962,  0.1962,  1.2089])
    >>> _extract_shape(tensor([0, 0, 1, 0]), tensor([[1., 2., 3., 4.],[0., 1., 2., 3.],[-1., 1., -1., 1.],[1., 2., 2., 3.]]), 0, tensor([-1.2089303, -0.19618238, 0.19618238, 1.2089303]))
    tensor([-1.1962, -0.2627,  0.2627,  1.1962])
    """
    _a = []
    # shift all time-series to minimize their distance from the current centroid
    for i in range(len(idx)):
        if idx[i] == j:
            # do not shift the signal if current centroid is the initial one filled with zeros
            if cur_center.sum() == 0:
                opt_x = x[i]
            else:
                _, opt_x = _sbd(cur_center, x[i])
            _a.append(opt_x)
    # no time-series for the cluster, the centroids is an array of zeros
    if len(_a) == 0:
        return torch.zeros_like(cur_center)

    a = stack(_a)

    columns = a.shape[1]
    y = zscore(a, axis=1, ddof=1)
    s = y.transpose(0, 1).mm(y)

    p = torch.empty((columns, columns), device=x.device, dtype=x.dtype)
    p.fill_(1.0 / columns)
    p = torch.eye(columns, device=x.device, dtype=x.dtype) - p

    m = p.mm(s).mm(p)
    _, vec = symeig(m, eigenvectors=True)
    centroid = vec[:, -1]
    finddistance1 = sqrt(torch.pow(a[0].sub(centroid), 2).sum())
    finddistance2 = sqrt(torch.pow(a[0].add(centroid), 2).sum())

    if finddistance1 >= finddistance2:
        centroid.mul_(-1)

    return zscore(centroid, ddof=1)


def _kshape_pytorch(x, k, max_iterations):
    """
    >>> # since PyTorch 0.3 version you only need to set torch.manual_seed which will seed all devices, including gpu-s
    >>> # from core import_kshape_pytorch
    >>> torch.manual_seed_all(0)
    >>> torch_device = torch.device("gpu")
    >>> x = torch.tensor([[1,2,3,4], [0,1,2,3], [-1,1,-1,1], [1,2,2,3]], device=torch_device)
    >>> _kshape_pytorch(x, k)
    (array([0, 0, 1, 0]), array([[-1.2244258 , -0.35015476,  0.52411628,  1.05046429],
           [-0.8660254 ,  0.8660254 , -0.8660254 ,  0.8660254 ]]))
    """
    # randomly assign time-series to one of k's clusters
    n = x.size(0)  # number of time-series (data points)
    len = x.size(1)  # number of samples per time-series (the length/width of the time-series)
    # idx - randomly assign each time-series to a cluster
    idx = torch.randint(0, k, (n,), device=x.device)
    # len: the lenght/width of the centroid is the same as the length/width of the time-series
    centroids = torch.zeros(k, len, device=x.device, dtype=x.dtype)
    # distances = torch.empty(m, k, device = torch_device)

    for _ in range(max_iterations):
        old_idx = idx
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])

        distances = (1 - _ncc_c_3dim(x, centroids).max(axis=2)).T

        idx = distances.argmin(1)
        if tensor_equal(old_idx, idx):
            break

    return idx, centroids


def kshape_pytorch(x, k, device="cpu", max_iterations=100):
    """
    Find k clusters for time-series in x.

    :param x: time series set X with n time-series, each of which is of length len
    :param k: required number of clusters
    :param device: "cpu" or "gpu"
    :param max_iterations: maximum number of iterations, where each iterations is composed of two steps:
    (1) assignment step: update the cluster membership for each time-series
    (2) refinement step: update the cluster centroids
    :return:
    """
    torch_device = torch.device(device)
    if isinstance(x, np.ndarray):
        # to avoid a copy of the data use from_numpy
        x = torch.from_numpy(x).to(torch_device)
    else:
        x = torch.tensor(x, device=torch_device)

    idx, centroids = _kshape_pytorch(x, k, torch_device, max_iterations=max_iterations)
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