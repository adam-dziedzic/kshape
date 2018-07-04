import unittest

import numpy as np
import torch

from kshape.core_pytorch import pytorch_conjugate, complex_mul_2dim

"""
Test the _ncc_c_3dim - most of the code here is repeated from the function to be able to verify each step and compare
it with the results from numpy.
"""


class NumpyPytorchNccTest3D(unittest.TestCase):

    def setUp(self):
        self.num_type = np.double
        self.x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.0, 0.0, 2.0]], dtype=self.num_type)
        self.centroids = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 2.0]], dtype=self.num_type)

    def test_norm(self):
        np_x_norm = np.linalg.norm(self.x, axis=1)
        print(np_x_norm)

        torch_x_norm = torch.norm(torch.from_numpy(self.x), p=2, dim=1)
        print(torch_x_norm)

        np.testing.assert_array_equal(np_x_norm, torch_x_norm.numpy())

    def test_norm_unsqeeze(self):
        np_x_norm = np.linalg.norm(self.x, axis=1)[:, None]
        print(np_x_norm)

        torch_x_norm = torch.norm(torch.from_numpy(self.x), p=2, dim=1).unsqueeze(-1)
        print(torch_x_norm)

        np.testing.assert_array_equal(np_x_norm, torch_x_norm.numpy())

    def compare_arrays2D(self, np_xfft, torch_xfft):
        print("shape of np_xfft: ", np_xfft.shape)
        print("shape of torch_xfft: ", torch_xfft.size())
        # torch returns only half of the fft - as the other half is conjugate symmetric (from 1 to half of the signal)
        for i in range(torch_xfft.shape[0]):
            for j in range(torch_xfft.shape[1]):
                # print(i, j)
                np.testing.assert_almost_equal(self.num_type(torch_xfft[i, j, 0].item()),
                                               self.num_type(np.real(np_xfft[i, j])), decimal=6)
                np.testing.assert_almost_equal(self.num_type(torch_xfft[i, j, 1].item()),
                                               self.num_type(np.imag(np_xfft[i, j])), decimal=6)

    def compare_arrays3D(self, np_xfft, torch_xfft):
        print("shape of np_xfft: ", np_xfft.shape)
        print("shape of torch_xfft: ", torch_xfft.size())
        # torch returns only half of the fft - as the other half is conjugate symmetric (from 1 to half of the signal)
        for i in range(torch_xfft.shape[0]):
            for j in range(torch_xfft.shape[1]):
                for k in range(torch_xfft.shape[2]):
                    # print(i, j, k)
                    np.testing.assert_almost_equal(self.num_type(torch_xfft[i, j, k, 0].item()),
                                                   self.num_type(np.real(np_xfft[i, j, k])), decimal=6)
                    np.testing.assert_almost_equal(self.num_type(torch_xfft[i, j, k, 1].item()),
                                                   self.num_type(np.imag(np_xfft[i, j, k])), decimal=6)

    def get_xfft(self):
        x_len = self.x.shape[-1]
        # bit_length - number of bits required to represent the expression (37).bit_length() return 6
        # it works like int(np.ceil(np.log2(37)))
        fft_size = 1 << (2 * x_len - 1).bit_length()
        pad_size = fft_size - x_len
        np_xfft = np.fft.fft(self.x, fft_size)
        print("np xfft: ", np_xfft)

        x = torch.from_numpy(self.x)
        padding = torch.zeros(self.x.shape[0], pad_size, device=x.device, dtype=x.dtype)
        x = torch.cat((x, padding), dim=1)
        signal_ndim = 1
        torch_xfft = torch.rfft(x, signal_ndim)
        print("torch xfft: ", torch_xfft)

        return np_xfft, torch_xfft

    def test_fft(self):
        np_xfft, torch_xfft = self.get_xfft()

        self.compare_arrays2D(np_xfft, torch_xfft)

    def get_cfft(self):
        c_len = self.centroids.shape[-1]
        fft_size = 1 << (2 * c_len - 1).bit_length()
        pad_size = fft_size - c_len
        np_cfft = np.conj(np.fft.fft(self.centroids, fft_size))
        print("np centroids fft: ", np_cfft)

        c = torch.from_numpy(self.centroids)
        padding = torch.zeros(self.centroids.shape[0], pad_size, device=c.device, dtype=c.dtype)
        # reverse the order of elements in the time domain what is equivalent to the conjugate in the frequency domain
        print("c before flipping: ", c)
        # c = flip2(c, dim=1)
        print("c after flipping: ", c)
        c = torch.cat((c, padding), dim=1)
        print("c after flipping: ", c)
        signal_ndim = 1
        torch_cfft = torch.rfft(c, signal_ndim)
        torch_cfft = pytorch_conjugate(torch_cfft)
        print("torch cfft: ", torch_cfft)

        return np_cfft, torch_cfft

    def test_fft_conjugate(self):
        np_cfft, torch_cfft = self.get_cfft()

        self.compare_arrays2D(np_cfft, torch_cfft)

    def get_muliplied_ffts(self):
        np_xfft, torch_xfft = self.get_xfft()
        np_cfft, torch_cfft = self.get_cfft()
        print("np_cfft: ", np_cfft)
        print("np_cfft shape: ", np_cfft.shape)
        print("np_cfft with extended dimensions: ", np_cfft[:, None])
        print("np_cfft with extended dimensions shape: ", np_cfft[:, None].shape)
        np_mul = np_xfft * np_cfft[:, None]
        print("np_mul: ", np_mul)
        print("torch cfft: ", torch_cfft)
        print("torch cfft size: ", torch_cfft.size())
        print("torch cfft with extended dimensions: ", torch_cfft.unsqueeze(-3))
        print("torch cfft with extended dimensions size: ", torch_cfft.unsqueeze(-3).size())
        torch_mul = complex_mul_2dim(torch_xfft, torch_cfft.unsqueeze(-3))
        print("torch_mul: ", torch_mul)
        return np_mul, torch_mul

    def test_complex_multiply(self):
        np_mul, torch_mul = self.get_muliplied_ffts()

        self.compare_arrays3D(np_mul, torch_mul)

    def get_irfft(self):
        x_len = self.x.shape[-1]
        fft_size = 1 << (2 * x_len - 1).bit_length()
        signal_ndim = 1

        np_mul, torch_mul = self.get_muliplied_ffts()

        np_cc = np.fft.ifft(np_mul)
        np_cc = np.concatenate((np_cc[:, :, -(x_len - 1):], np_cc[:, :, :x_len]), axis=2)
        np_cc = np.real(np_cc)
        print("np_cc: ", np_cc)

        torch_cc = torch.irfft(torch_mul, signal_ndim=signal_ndim, signal_sizes=(fft_size,))
        torch_cc = torch.cat((torch_cc[:, :, -(x_len - 1):], torch_cc[:, :, :x_len]), dim=2)
        print("torch_cc: ", torch_cc)

        return np_cc, torch_cc

    def test_irfft(self):
        np_cc, torch_cc = self.get_irfft()

        self.compare_arrays2D(np_cc, torch_cc)

    def get_denominator(self):
        den = np.linalg.norm(self.x, axis=1)[:, None] * np.linalg.norm(self.centroids, axis=1)
        den[den == 0] = np.Inf
        np_den = den.T[:, :, None]
        print("np_den: ", np_den)

        x = torch.from_numpy(self.x)
        c = torch.from_numpy(self.centroids)
        den = torch.norm(x, p=2, dim=1).unsqueeze(-1) * torch.norm(c, p=2, dim=1)
        den[den == 0] = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        torch_den = den.transpose(0, 1).unsqueeze(-1)
        print("torch_den: ", torch_den)

        return np_den, torch_den

    def test_den(self):
        """
        Test the denominator of _ncc_c_3dim().
        """
        np_den, torch_den = self.get_denominator()
        np.testing.assert_array_almost_equal(np_den, torch_den.numpy())

    def result_ncc_c_3dim(self):
        x_len = self.x.shape[-1]
        np_cc, torch_cc = self.get_irfft()
        np_den, torch_den = self.get_denominator()

        np_ncc = np_cc / np_den
        print("numpy_ncc_result: ", np_ncc)

        torch_ncc = torch.div(torch_cc, torch_den)
        print("torch_ncc_result: ", torch_ncc)

        self.compare_arrays3D(np_ncc, torch_ncc)


if __name__ == '__main__':
    unittest.main()
