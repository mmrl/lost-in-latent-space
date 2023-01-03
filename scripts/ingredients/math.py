import math
import numpy as np


def _pair(s):
    if not isinstance(s, tuple):
        return s, s
    return s


def maxpool2d_out_shape(in_shape, pool_shape, stride, padding):
    in_channels, hout, wout = in_shape
    pool_shape = _pair(pool_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(pool_shape, stride, padding)

    hout = math.floor((hout - hval[0] + 2 * hval[2]) / hval[1]) + 1
    wout = math.floor((wout - wval[0] + 2 * wval[2]) / wval[1]) + 1

    return in_channels, hout, wout


def conv2d_out_shape(in_shape, out_channels, kernel_shape, stride, padding):
    if isinstance(padding, str):
        if padding == 'same':
            return out_channels, *in_shape[1:]
        elif padding == 'valid':
            padding = 0
        else:
            ValueError('Unrecognized padding value {}'.format(padding))

    in_shape = in_shape[1:]
    kernel_shape = _pair(kernel_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(in_shape, kernel_shape, stride, padding)

    hout = math.floor((hval[0] - hval[1] + 2 * hval[3]) / hval[2]) + 1
    wout = math.floor((wval[0] - wval[1] + 2 * wval[3]) / wval[2]) + 1

    return out_channels, hout, wout


def transp_conv2d_out_shape(in_shape, out_channels, kernel_shape,
                            stride, padding):
    in_shape = in_shape[1:]
    kernel_shape = _pair(kernel_shape)
    stride = _pair(stride)
    padding = _pair(padding)

    hval, wval = zip(in_shape, kernel_shape, stride, padding)

    hout = (hval[0] - 1) * hval[2] - 2 * hval[3] + hval[1]
    wout = (wval[0] - 1) * wval[2] - 2 * wval[3] + wval[1]

    return out_channels, hout, wout


def compute_flattened_size(input_size, start_dim=1, end_dim=-1):
    start_dim -= 1
    if start_dim < 0:
        raise ValueError('Cannot flatten batch dimension')

    if end_dim < 0:
        end_dim = len(input_size) + 1

    output_size = list(input_size[:start_dim])
    output_size.append(np.prod(input_size[start_dim:end_dim]))
    output_size.extend(input_size[end_dim:])

    if len(output_size) == 1:
        return output_size[0]

    return output_size

