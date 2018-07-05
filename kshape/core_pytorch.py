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
    >>> result = flip2(tensor([1, 2, 3]), dim=0)
    >>> np.testing.assert_array_equal(result, tensor([3, 2, 1]))
    """
    return x.flip([dim])


def flip2(x, dim=0):
    """
    Flip the tensor x for dimension dim.

    :param x: the input tensor
    :param dim: the dimension according to which we flip the tensor
    :return: flipped tensor

    This flip method is used only for version of PyTorch <= 0.4. There is a flip method added to Tensor in PyTorch 5.0.

    >>> result = flip2(tensor([1, 2, 3]), dim=0)
    >>> np.testing.assert_array_equal(result, tensor([3, 2, 1]))
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


# flip = flip1
flip = flip2  # I suppose there is a bug in current flip function in PyTorch so we revert to our own version of flip

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

    >>> result = zscore(tensor([1., 2., 3.]))
    >>> np.testing.assert_array_almost_equal(result, tensor([-1.2247,  0.0000,  1.2247]), decimal=4)
    >>> result = zscore(tensor([1., 2., 3.]), ddof=1)
    >>> np.testing.assert_array_almost_equal(result, tensor([-1.,  0.,  1.]), decimal=4)
    >>> result = zscore(tensor([[1.,2.,3.],[4.,5.,6.]]), axis=1)
    >>> np.testing.assert_array_almost_equal(result, tensor([[-1.2247,  0.0000,  1.2247], [-1.2247,  0.0000,  1.2247]]), decimal=4)
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
    :return: shifted time-series

    >>> result = roll_zeropad(tensor([1., 2., 3.]), shift=2)
    >>> np.testing.assert_array_equal(result, tensor([0., 0., 1.]))
    >>> result=roll_zeropad(tensor([1, 2, 3]), 0)
    >>> np.testing.assert_array_equal(result, tensor([1, 2, 3]))
    >>> result=roll_zeropad(tensor([1., 2., 3.]), -1)
    >>> np.testing.assert_array_equal(result, tensor([2., 3., 0.]))
    >>> result=roll_zeropad(tensor([1., 2., 3.]), 3)
    >>> np.testing.assert_array_equal(result, tensor([0., 0., 0.]))
    >>> result=roll_zeropad(tensor([1, 2, 3]), 4)
    >>> np.testing.assert_array_equal(result, tensor([0, 0, 0]))
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
    >>> # torch.equal(tensor1, tensor2): True if two tensors have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul(x, y), tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))
    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))
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


def complex_mul_2dim(x, y):
    """
    Multiply arrays of complex numbers. Each complex number is expressed as a pair of real and imaginary parts.

    :param x: the first 2D (two-dimensional) array of complex numbers
    :param y: the second 2D (two-dimensional) array complex numbers
    :return: result of multiplication (an array with complex numbers)
    # based on the paper: Fast Algorithms for Convolutional Neural Networks (https://arxiv.org/pdf/1509.09308.pdf)
    # based on the paper: Fast Algorithms for Convolutional Neural Networks (https://arxiv.org/pdf/1509.09308.pdf)
    >>> # x = torch.rfft(torch.tensor([1., 2., 3., 0.]), 1)
    >>> x = tensor([[ 6.,  0.], [-2., -2.], [ 2.,  0.]])
    >>> # y = torch.rfft(torch.tensor([5., 6., 7., 0.]), 1)
    >>> y = tensor([[18.,  0.], [-2., -6.], [ 6.,  0.]])
    >>> # torch.equal(tensor1, tensor2): True if two tensors have the same size and elements, False otherwise.
    >>> np.testing.assert_array_equal(complex_mul_2dim(x, y), tensor([[108.,   0.], [ -8.,  16.], [ 12.,   0.]]))
    >>> x = tensor([[1., 2.]])
    >>> y = tensor([[2., 3.]])
    >>> xy = complex_mul_2dim(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[-4., 7.]]))
    >>> x = tensor([[[1., 2.]]])
    >>> y = tensor([[[2., 3.]]])
    >>> xy = complex_mul_2dim(x, y)
    >>> np.testing.assert_array_equal(xy, tensor([[[-4., 7.]]]))
    """
    ua = x[..., 0]
    va = y[..., 0]
    ub = x[..., 0] + x[..., 1]
    vb = y[..., 1]
    uc = x[..., 1] - x[..., 0]
    vc = y[..., 0] + y[..., 1]
    uavc = mul(ua, vc)
    ucva = mul(uc, va)
    # print("uavc shape: ", uavc.shape)
    # print("ucva shape: ", ucva.shape)
    # print("shape x: ", x.shape)
    # print("shape of add(ucva, uavc): ", add(ucva, uavc).shape)
    result = torch.empty(*ucva.shape, 2, dtype=x.dtype, device=x.device)
    result[..., 1] = add(ucva, uavc)
    result[..., 0] = add(uavc, mul(mul(ub, vb), -1))
    return result


def pytorch_conjugate(x):
    """
    Conjugate all the complex numbers in tensor x in place.

    :param x: PyTorch tensor with complex numbers
    :return: conjugated numbers in x

    >>> x = tensor([[1, 2]])
    >>> x = pytorch_conjugate(x)
    >>> np.testing.assert_array_equal(x, tensor([[1, -2]]))
    """
    x[..., 1].mul_(-1)
    return x


def _ncc_c_torch(x, y):
    """
    # Variant of NCCc that operates with 1 dimensional x array and 1 dimensional y array.
    :param x: one-dimensional array
    :param y: one-dimensional array
    :return: normalized cross correlation (with coefficient normalization)

    >>> result1 = _ncc_c_torch(tensor([1.,2.,3.,4.]), tensor([1.,2.,3.,4.]))
    >>> expected1 = tensor([0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333])
    >>> np.testing.assert_array_almost_equal(result1, expected1, decimal=4)
    >>> result2 = _ncc_c_torch(tensor([1.,1.,1.]), tensor([1.,1.,1.]))
    >>> expected2 = tensor([0.3333, 0.6667, 1.0000, 0.6667, 0.3333])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c_torch(tensor([1.,2.,3.]), tensor([-1.,-1.,-1.]))
    >>> expected3 = tensor([-0.1543, -0.4629, -0.9258, -0.7715, -0.4629])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    >>> result4 = _ncc_c_torch(tensor([1.1,5.5,3.9,1.0]), tensor([9.1,-1.1,0.7,-1.3]))
    >>> expected4 = tensor([-0.0223, -0.0995, -0.0379, 0.0841, 0.7248, 0.5365, 0.142])
    >>> np.testing.assert_array_almost_equal(result4, expected4, decimal=4)
    """
    # the denominator for normalization
    den = norm(x) * norm(y)
    if den == 0:
        den = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    # print("x view", x.view(1,1,-1))
    # x_len = len(x)
    # fft_size = 1 << (2 * x_len - 1).bit_length()
    # pad_size = fft_size - x_len
    # padding = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
    # x = cat((x, padding))
    cc = torch.nn.functional.conv1d(x.view(1, 1, -1), y.view(1, 1, -1), padding=x.shape[-1] - 1).squeeze()
    return div(cc, den)


def _ncc_c(x, y):
    """
    # Variant of NCCc that operates with 1 dimensional x array and 1 dimensional y array.
    :param x: one-dimensional array
    :param y: one-dimensional array
    :return: normalized cross correlation (with coefficient normalization)

    >>> result1 = _ncc_c(tensor([1.,2.,3.,4.]), tensor([1.,2.,3.,4.]))
    >>> expected1 = tensor([0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333])
    >>> np.testing.assert_array_almost_equal(result1, expected1, decimal=4)
    >>> result2 = _ncc_c(tensor([1.,1.,1.]), tensor([1.,1.,1.]))
    >>> expected2 = tensor([0.3333, 0.6667, 1.0000, 0.6667, 0.3333])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c(tensor([1.,2.,3.]), tensor([-1.,-1.,-1.]))
    >>> expected3 = tensor([-0.1543, -0.4629, -0.9258, -0.7715, -0.4629])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    >>> result4 = _ncc_c(tensor([1.1,5.5,3.9,1.0]), tensor([9.1,-1.1,0.7,-1.3]))
    >>> expected4 = tensor([-0.0223, -0.0995, -0.0379, 0.0841, 0.7248, 0.5365, 0.142])
    >>> np.testing.assert_array_almost_equal(result4, expected4, decimal=4)
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
    # yfft = pytorch_conjugate(yfft)
    cc = irfft(complex_mul(xfft, yfft), signal_ndim=signal_ndim, signal_sizes=(fft_size,))
    return div(cc[:(2 * x_len - 1)], den)


def _ncc_c_2dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional X arrays and 1 dimensional
    y vector.

    :param x: 2 dimensional array with time series
    :param y: 1 dimensional array with a single centroid
    :return: a 2 dimensional array of normalized fourier transforms

    >>> result1 = _ncc_c_2dim(tensor([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]]), tensor([1.0, 2.0, 1.0]))
    >>> expected1 = tensor([[0.10910895, 0.43643578, 0.87287156, 0.87287156, 0.32732684], [0.35634832, 0.80178373, 0.71269665, 0.4454354 , 0.17817416]])
    >>> np.testing.assert_array_almost_equal(result1, expected1, decimal=4)
    >>> result1 = _ncc_c_2dim(tensor([[1.,2.,3.,4.]]), tensor([1.,2.,3.,4.]))
    >>> expected1 = tensor([[0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333]])
    >>> np.testing.assert_array_almost_equal(result1, expected1, decimal=4)
    >>> result2 = _ncc_c_2dim(tensor([[1.,1.,1.]]), tensor([1.,1.,1.]))
    >>> expected2 = tensor([[0.3333, 0.6667, 1.0000, 0.6667, 0.3333]])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c_2dim(tensor([[1.,2.,3.]]), tensor([-1.,-1.,-1.]))
    >>> expected3 = tensor([[-0.1543, -0.4629, -0.9258, -0.7715, -0.4629]])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    """
    # the denominator for normalization
    den = norm(x, p=2, dim=1) * norm(y)
    # for array den with values 0, replaces 0 values in den with float('inf')
    # https://goo.gl/XSxvau : boolean indexing in Python
    den[den == 0] = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    signal_ndim = 1
    x_len = x.shape[-1]
    fft_size = 1 << (2 * x_len - 1).bit_length()
    pad_size = fft_size - x_len
    x_padding = torch.zeros(x.shape[0], pad_size, device=x.device, dtype=x.dtype)
    y_padding = torch.zeros(pad_size, device=y.device, dtype=y.dtype)
    # conjugate in the frequency domain is equivalent to the reversed signal in the time domain
    y = flip(y, dim=0)
    x = cat((x, x_padding), dim=1)
    y = cat((y, y_padding))
    xfft = rfft(x, signal_ndim)
    yfft = rfft(y, signal_ndim)
    # yfft = pytorch_conjugate(yfft)
    cc = irfft(complex_mul_2dim(xfft, yfft), signal_ndim=signal_ndim, signal_sizes=(fft_size,))
    return div(cc[:, :(2 * x_len - 1)], den.unsqueeze(-1))


def _ncc_c_3dim_torch(x, y):
    """
    Variant of NCCc that operates with 2 dimensional x arrays and 2 dimensional y arrays.
    Returns a 3 dimensional array of normalized fourier transforms.

    >>> result = _ncc_c_3dim_torch(tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 0.0, 2.0]]), tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 2.0]]))
    >>> expected = tensor([[[0.4629, 0.7715, 0.9258, 0.4629, 0.1543], [0.3948, 0.7237, 0.9869, 0.5922, 0.2632], [0.5164, 0.5164, 0.7746, 0.2582, 0.2582]], [[0.3586, 0.2390, 0.8367, 0.4781, 0.2390], [0.3058, 0.2548, 0.8154, 0.5096, 0.4077], [0.4000, 0.0000, 1.0000, 0.0000, 0.4000]]])
    >>> np.testing.assert_array_almost_equal(result, expected, decimal=4)
    >>> result2 = _ncc_c_3dim_torch(tensor([[1.,2.,3.]]), tensor([[-1.,-1.,-1.]]))
    >>> expected2 = tensor([[[-0.4629, -0.7715, -0.9258, -0.4629, -0.1543]]])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c_3dim_torch(tensor([[1.,2.,3.,4.]]), tensor([[1.,2.,3.,4.]]))
    >>> expected3 = tensor([[[0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333]]])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    >>> result4 = _ncc_c_3dim_torch(tensor([[1.,1.,1.]]), tensor([[1.,1.,1.]]))
    >>> expected4 = tensor([[[0.3333, 0.6667, 1.0000, 0.6667, 0.3333]]])
    >>> np.testing.assert_array_almost_equal(result4, expected4, decimal=4)
    """
    # Apply the L2 norm (the p=2 - the exponent value in the norm formulation).
    den = torch.mul(norm(x, p=2, dim=1).unsqueeze(-1), norm(y, p=2, dim=1))
    # for array den with values 0, replaces 0 values in den with float('inf')
    # https://goo.gl/XSxvau : boolean indexing in Python
    den[den == 0] = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    cc = torch.nn.functional.conv1d(y.unsqueeze(1), x.unsqueeze(1), padding=y.shape[-1] - 1)
    den = den.transpose(0, 1).unsqueeze(-1)
    result = div(cc, den)
    return result


def _ncc_c_3dim(x, y):
    """
    Variant of NCCc that operates with 2 dimensional x arrays and 2 dimensional y arrays.
    Returns a 3 dimensional array of normalized fourier transforms.

    >>> result = _ncc_c_3dim(tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 0.0, 2.0]]), tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 2.0]]))
    >>> expected = tensor([[[0.1543, 0.4629, 0.9258, 0.7715, 0.4629], [0.2632, 0.5922, 0.9869, 0.7237, 0.3948], [0.2582, 0.2582, 0.7746, 0.5164, 0.5164]], [[0.2390, 0.4781, 0.8367, 0.2390, 0.3586], [0.4077, 0.5096, 0.8154, 0.2548, 0.3058], [0.4000, 0.0000, 1.0000, 0.0000, 0.4000]]])
    >>> np.testing.assert_array_almost_equal(result, expected, decimal=4)
    >>> result2 = _ncc_c_3dim(tensor([[1.,2.,3.]]), tensor([[-1.,-1.,-1.]]))
    >>> expected2 = tensor([[[-0.1543, -0.4629, -0.9258, -0.7715, -0.4629]]])
    >>> np.testing.assert_array_almost_equal(result2, expected2, decimal=4)
    >>> result3 = _ncc_c_3dim(tensor([[1.,2.,3.,4.]]), tensor([[1.,2.,3.,4.]]))
    >>> expected3 = tensor([[[0.1333, 0.3667, 0.6667, 1.0000, 0.6667, 0.3667, 0.1333]]])
    >>> np.testing.assert_array_almost_equal(result3, expected3, decimal=4)
    >>> result4 = _ncc_c_3dim(tensor([[1.,1.,1.]]), tensor([[1.,1.,1.]]))
    >>> expected4 = tensor([[[0.3333, 0.6667, 1.0000, 0.6667, 0.3333]]])
    >>> np.testing.assert_array_almost_equal(result4, expected4, decimal=4)
    """
    # Apply the L2 norm (the p=2 - the exponent value in the norm formulation).
    den = norm(x, p=2, dim=1).unsqueeze(-1) * norm(y, p=2, dim=1)
    # for array den with values 0, replaces 0 values in den with float('inf')
    # https://goo.gl/XSxvau : boolean indexing in Python
    den[den == 0] = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    signal_ndim = 1
    x_len = x.shape[-1]
    y_len = y.shape[-1]
    assert x_len == y_len  # the centroids should be of the same length as time-series
    fft_size = 1 << (2 * x_len - 1).bit_length()
    pad_size = fft_size - x_len
    x_padding = torch.zeros(x.shape[0], pad_size, device=x.device, dtype=x.dtype)
    y_padding = torch.zeros(y.shape[0], pad_size, device=x.device, dtype=x.dtype)
    # conjugate in the frequency domain is equivalent to the reversed signal in the time domain
    y = flip(y, dim=1)
    x = cat((x, x_padding), dim=1)
    y = cat((y, y_padding), dim=1)
    xfft = rfft(x, signal_ndim)
    yfft = rfft(y, signal_ndim)
    # yfft = pytorch_conjugate(yfft)
    # use broadcasting ...unsqueeze(-3) to compute the distances for each pair of time-series and centroids
    cc = irfft(complex_mul_2dim(xfft, yfft.unsqueeze(-3)), signal_ndim=signal_ndim, signal_sizes=(fft_size,))
    den = den.transpose(0, 1).unsqueeze(-1)
    cc = cc[:, :, :(2 * x_len - 1)]
    result = div(cc, den)
    return result


def _sbd(x, y):
    """
    Shape based distance between x and y.

    :param x: the first time-series
    :param y: the seconda time-series
    :return: shape based distance between x and y, shifted y to the position that maximizes the similarity of x and y

    >>> dist, y = _sbd(tensor([1.,1.,1.]), tensor([1.,1.,1.]))
    >>> np.testing.assert_array_almost_equal(dist, tensor(0.), decimal=4)
    >>> np.testing.assert_array_equal(y, tensor([1., 1., 1.]))
    >>> dist, y = _sbd(tensor([0.,1.,2.]), tensor([1.,2.,3.]))
    >>> np.testing.assert_array_almost_equal(dist, tensor(0.0438), decimal=4)
    >>> np.testing.assert_array_equal(y, tensor([1., 2., 3.]))
    >>> dist, y = _sbd(tensor([1.,2.,3.]), tensor([0.,1.,2.]))
    >>> np.testing.assert_array_almost_equal(dist, tensor(0.0438), decimal=4)
    >>> np.testing.assert_array_equal(y, tensor([0., 1., 2.]))
    >>> dist, y = _sbd(tensor([1.,2.,3.]), tensor([-1.,-1.,-1.]))
    >>> np.testing.assert_array_almost_equal(dist, tensor(1.1543), decimal=4)
    >>> np.testing.assert_array_equal(y, tensor([-1.,  0.,  0.]))
    >>> dist, y = _sbd(tensor([1.,2.,3.], dtype=torch.float64), tensor([0.,1.,2.], dtype=torch.float64))
    >>> np.testing.assert_array_almost_equal(dist, tensor(0.0438, dtype=torch.float64), decimal=4)
    >>> np.testing.assert_array_equal(y, tensor([0., 1., 2.], dtype=torch.float64))
    >>> dist, y = _sbd(tensor([0., 0., 1., 2., 3., 0., 0.]), tensor([1., 2., 3., 0., 0., 0., 0.]))
    >>> np.testing.assert_array_almost_equal(dist, tensor(5.9605e-08), decimal=4)
    >>> np.testing.assert_array_equal(y, tensor([0., 0., 1., 2., 3., 0., 0.]))
    """
    ncc = _ncc_c(x, y)
    idx = ncc.argmax().item()
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

    >>> result = _extract_shape(tensor([0, 1]), tensor([[1., 2., 3.], [4., 5., 6.]]), 2, tensor([0., 3., 4.]))
    >>> np.testing.assert_array_almost_equal(result, tensor([0., 0., 0.]), decimal=4)
    >>> result = _extract_shape(tensor([0, 1]), tensor([[1., 2., 3.], [4., 5., 6.]]), 1, tensor([0., 3., 4.]))
    >>> np.testing.assert_array_almost_equal(result, tensor([-1.,  0.,  1.]), decimal=4)
    >>> result = _extract_shape(tensor([0, 1]), tensor([[-1., 2., 3.], [4., -5., 6.]]), 1, tensor([0., 3., 4.]))
    >>> np.testing.assert_array_almost_equal(result, tensor([-0.9684,  1.0289, -0.0605]), decimal=4)
    >>> result = _extract_shape(tensor([1, 0, 1, 0]), tensor([[1., 2., 3., 4.], [0., 1., 2., 3.], [-1., 1., -1., 1.], [1., 2., 2., 3.]]), 0, tensor([0., 0., 0., 0.]))
    >>> np.testing.assert_array_almost_equal(result, tensor([-1.2089, -0.1962,  0.1962,  1.2089]), decimal=4)
    >>> result = _extract_shape(tensor([0, 0, 1, 0]), tensor([[1., 2., 3., 4.],[0., 1., 2., 3.],[-1., 1., -1., 1.],[1., 2., 2., 3.]]), 0, tensor([-1.2089303, -0.19618238, 0.19618238, 1.2089303]))
    >>> np.testing.assert_array_almost_equal(result, tensor([-1.1962, -0.2627,  0.2627,  1.1962]), decimal=4)
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


def _kshape_pytorch(x, k, max_iterations=100, idx=None):
    """
    The main call of kshape.

    :param x: the 2 dimensional array with time-series data
    :param k: the scalar with number of expected clusters
    :param max_iterations: how many times iterate through the time-series data, where each iteration is composed of two
    steps: 1) cluster membership assignment, 2) centroid computation
    :param idx: the initial assignment of time series to clusters
    :return: a two element tuple where at the first position we have (for each time series) - its index of a cluster
    and, in the second element of the tuple, the centroids for each cluster.

    >>> # since PyTorch 0.3 version you only need to set torch.manual_seed which will seed all devices, including gpu-s
    >>> # from core import_kshape_pytorch
    >>> # torch.manual_seed(0) # no need to set the seed - we set the initial cluster assignment

    >>> result_cluster_assignment, result_centroids = _kshape_pytorch(tensor([[1.0,2.0,3.0,4.0], [0.0,1.0,2.0,3.0], [-1.0,1.0,-1.0,1.0], [1.0,2.0,2.0,3.0], [1.0,2.2,-2.0,-3.0], [-1.1,2.3,-2.9,3.4]], dtype=torch.double), 3, idx=torch.tensor([1, 2, 1, 0, 0, 1]))
    >>> expected_cluster_assignments, expected_centroids = (tensor([2, 2, 1, 0, 0, 1]), tensor([[-0.663535, -1.008225,  0.565868,  1.105892], [-0.701075,  0.761482, -1.011736,  0.95133 ], [-1.161895, -0.387298,  0.387299,  1.161895]]))
    >>> np.testing.assert_array_equal(result_cluster_assignment, expected_cluster_assignments)
    >>> np.testing.assert_array_almost_equal(result_centroids, expected_centroids)

    >>> torch_device = torch.device("cpu")
    >>> result_cluster_assignment, result_centroids = _kshape_pytorch(tensor([[1.0,2.0,3.0,4.0], [0.0,1.0,2.0,3.0], [-1.0,1.0,-1.0,1.0], [1.0,2.0,2.0,3.0]]), 2, idx=torch.tensor([1, 0, 1, 0]))
    >>> expected_cluster_assignments, expected_centroids = (tensor([0, 0, 1, 0]), tensor([[-1.050464, -0.524116,  0.350155,  1.224426], [-0.866025,  0.866025, -0.866025,  0.866025]]))
    >>> np.testing.assert_array_equal(result_cluster_assignment, expected_cluster_assignments)
    >>> np.testing.assert_array_almost_equal(result_centroids, expected_centroids)
    """
    # randomly assign time-series to one of k's clusters
    n = x.size(0)  # number of time-series (data points)
    len = x.size(1)  # number of samples per time-series (the length/width of the time-series)
    # idx - one-dimensional array of randomly assigned time-series to clusters
    if idx is None:
        idx = torch.randint(0, k, (n,), dtype=torch.long)
        idx = idx.to(x.device)
    # len: the lenght/width of the centroid is the same as the length/width of the time-series
    centroids = torch.zeros(k, len, device=x.device, dtype=x.dtype)
    # distances = torch.empty(m, k, device = torch_device)

    for _ in range(max_iterations):
        old_idx = idx.clone()
        for j in range(k):
            centroids[j] = _extract_shape(idx, x, j, centroids[j])

        distances = (1 - _ncc_c_3dim_torch(x, centroids).max(dim=2)[0]).transpose(0, 1)
        # # similarities = _ncc_c_3dim(x, centroids)
        # # # tensor.max in PyTorch returns a tuple. The first return element in the tuple is the maximum value of each
        # # # row of the input tensor in the given dimension dim. The second return value is the index location of each
        # # # maximum value found (argmax).
        # # max_similarities = similarities.max(dim=-1)[0]
        # # distances = (1 - max_similarities).transpose(0, 1)
        idx = distances.argmin(dim=1)

        # compute distance in a for loop - it works better for huge amount of data:
        # http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
        # for i, time_series in enumerate(x):
        #     similarities = _ncc_c_2dim(centroids, time_series)
        #     max_similarities = similarities.max(dim=-1)[0]
        #     closest_centroid_index = max_similarities.argmax().item()
        #     idx[i] = closest_centroid_index

        if torch.equal(old_idx, idx):
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
    :return: for each cluster returns its centroid and indexes of the time-series that belong to the cluster
    """
    torch_device = torch.device(device)
    if isinstance(x, np.ndarray):
        # to avoid a copy of the data use from_numpy
        x = torch.from_numpy(x).to(torch_device)
    else:
        x = torch.tensor(x, device=torch_device)

    idx, centroids = _kshape_pytorch(x, k, max_iterations=max_iterations)
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
