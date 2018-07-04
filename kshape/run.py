import core
import core_python
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--k", default=2, type=int, help="number of clusters")
parser.add_argument("--n", default=100, type=int, help="number of time-series")
parser.add_argument("--m", default=100, type=int, help="length of time-series")
parser.add_argument("--f", default=100, type=string, help="framework: numpy or torch")
parser.add_argument("--d", default=100, type=int, help="framework: numpy or torch")

args = parser.parse_args(sys.argv[1:])
print("number of clusters: ", args.k)
