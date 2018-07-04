import sys

import argparse
import numpy as np

from kshape import core_pytorch
from kshape import core

gpu = "gpu"
cpu = "cpu"
torch = "torch"
numpy = "numpy"
device_help = "select device: {} or {}".format(cpu, gpu)
framework_help = "select framework: {} or {}".format(numpy, torch)
datatype_help = "select numpy data type"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clusters", default=2, type=int, help="number of clusters")
parser.add_argument("-n", "--number", default=100, type=int, help="number of time-series")
parser.add_argument("-l", "--length", default=128, type=int, help="length of time-series")
parser.add_argument("-f", "--framework", default=numpy, help=framework_help)
parser.add_argument("-d", "--device", default=cpu, help=device_help)
parser.add_argument("-t", "--type", default="float32", help=datatype_help)

args = parser.parse_args(sys.argv[1:])
print("number of clusters: ", args.clusters)
print("number of time-series: ", args.number)
print("length of time-series: ", args.length)
print("selected framework: ", args.framework)
print("selected device: ", args.device)
print("selected data type: ", args.type)

x = np.random.rand(args.number, args.length)

try:
    x = np.ndarray.astype(x, dtype=np.dtype(type))
except TypeError as err:
    print(err)
    print("Error: ", datatype_help, " got: ", args.type)
    sys.stdout.flush()
    exit(1)

result = None
if args.device == gpu:
    result = core_pytorch.kshape_pytorch(x=x, k=args.clusters, device=gpu)
elif args.device == cpu:
    if args.framework == torch:
        result = core_pytorch.kshape_pytorch(x=x, k=args.clusters, device=cpu)
    elif args.framework == numpy:
        result = core.kshape(x=x, k=args.clusters)
else:
    print(device_help)

if result:
    print(result)
