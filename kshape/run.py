import sys
import time

import argparse
import numpy as np

from kshape import core
from kshape import core_pytorch
from kshape.data import load_time_series

gpu = "gpu"
cpu = "cpu"
torch = "torch"
numpy = "numpy"
device_help = "select device: {} or {}".format(cpu, gpu)
framework_help = "select framework: {} or {}".format(numpy, torch)
datatype_help = "select numpy data type"
time_series_number = "number of random time-series (if chosen)"
time_series_length = "length of random time-series (if chosen)"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--clusters", default=3, type=int, help="number of clusters")
parser.add_argument("-n", "--number", default=1000, type=int, help=time_series_number)
parser.add_argument("-l", "--length", default=1024, type=int, help=time_series_length)
parser.add_argument("-f", "--framework", default=numpy, help=framework_help)
parser.add_argument("-d", "--device", default=cpu, help=device_help)
parser.add_argument("-t", "--type", default="float64", help=datatype_help)
parser.add_argument("-s", "--sourcedata", default="StarLightCurves",
                    help="Choose a UCR time-series dataset or type 'random' to randomly generate data")

args = parser.parse_args(sys.argv[1:])
print("number of clusters: ", args.clusters)
print(time_series_number, ": ", args.number)
print(time_series_length, ": ", args.length)
print(framework_help, ": ", args.framework)
print("selected device: ", args.device)
print("selected data type: ", args.type)
print("selected source of data: ", args.sourcedata)

if args.sourcedata == "random":
    x = np.random.rand(args.number, args.length)
else:
    datasets = load_time_series.load_data(args.sourcedata)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    x = np.vstack((train_set_x, valid_set_x, test_set_x))
    print("loaded ", x.shape[0], " data points of length ", x.shape[1])

try:
    x = x.astype(dtype=args.type, copy=False)
except TypeError as err:
    print(err)
    print("Error: ", datatype_help, " got: ", args.type)
    sys.stdout.flush()
    exit(1)

result = None
start = time.time()
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
    print("elapsed time, ", time.time() - start, ",sec")
