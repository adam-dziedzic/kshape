# k-Shape

[![Build Status](https://travis-ci.org/Mic92/kshape.svg?branch=master)](https://travis-ci.org/Mic92/kshape)

Python implementation of [k-Shape](http://www.cs.columbia.edu/~jopa/kshape.html),
a new fast and accurate unsupervised time-series cluster algorithm.
[See also](#relevant-articles)

We used this implementation for our paper: [Sieve: Actionable Insights from Monitored Metrics in Distributed Systems](https://sieve-microservices.github.io/)

## Installation

kshape is available on PyPI https://pypi.python.org/pypi/kshape

```console
$ pip install kshape
```

### Install from source

If you are using a [virtualenv](https://virtualenv.pypa.io/en/stable/) activate it. Otherwise you can install
into the system python

```console
$ python setup.py install
```

## Usage

```python
from kshape.core import kshape, zscore

time_series = [[1,2,3,4], [0,1,2,3], [0,1,2,3], [1,2,2,3]]
cluster_num = 2
clusters = kshape(zscore(time_series, axis=1), cluster_num)
#=> [(array([-1.161895  , -0.38729833,  0.38729833,  1.161895  ]), [0, 1, 2]),
#    (array([-1.22474487,  0.        ,  0.        ,  1.22474487]), [3])]
```

Returns list of tuples with the clusters found by kshape. The first value of the
tuple is zscore normalized centroid. The second value of the tuple is the index
of assigned series to this cluster.
The results can be examined by drawing graphs of the zscore normalized values
and the corresponding centroid.

## Gotchas when working with real-world time series

- If the data is available from different sources with same frequency but at different points in time, it needs to be aligned.
- In the following a tab seperated file is assumed, where each column is a different observation;
  gapps in columns happen, when only a certain value at this point in time was obtained.

```python
import pandas as pd
# assuming the time series are stored in a tab seperated file, where `time` is
# the name of the column containing the timestamp
df = pd.read_csv(filename, sep="\t", index_col='time', parse_dates=True)
# use a meaningful sample size depending on how the frequency of your time series:
# Higher is more accurate, but if series gets too long, the calculation gets cpu and memory intensive.
# Keeping the length below 2000 values is usually a good idea.
df = df.resample("500ms").mean()
df.interpolate(method="time", limit_direction="both", inplace=True)
df.fillna(method="bfill", inplace=True)
```

- kshape also expect no time series with a constant observation value or 'n/a'

```python
time_series = []
for f in df.columns:
  if not df[f].isnull().any() and df[f].var() != 0:
    time_series.append[df[f]]
```

## Running on GPU
**Run the kshape algorithm on a GPU using the PyTorch library.**
Author: Adam Dziedzic

### Usage

```python
from kshape.core_gpu import kshape_gpu, zscore_gpu

time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
cluster_num = 2
clusters = kshape_gpu(zscore_gpu(time_series), cluster_num)
print("centroids and clusters: ", clusters)
second_centroid = clusters[1][0].numpy()
print("second centroid: ", second_centroid)
# can return (there is some randomness involved in the algorithm so this can differ): [(tensor([-1.2511,  1.3528, -0.5106,  0.5652, -0.1564]), [3]), (tensor([-1.3289, -0.8265,  0.7324,  0.6802,  0.7428]), [0, 1, 2])]
```

Returns list of tuples with the clusters found by kshape. The first value of the
tuple is zscore normalized centroid. The second value of the tuple is the index
of assigned series to this cluster.
The results can be examined by drawing graphs of the zscore normalized values
and the corresponding centroid.

### Settings
1. Data sets from UCR time-series archive or randomly generated with uniform distribution.
2. CPU 1: Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz, 8 cores, 32 GB RAM
3. GPU 1: Tesla P100 16 GB, 3584 cores
4. CPU 2: Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz, 48 cores, 128 GB RAM

| Device| Framework  | Dataset  | Time(sec) |Type|# of clusters|Comment|
| ------|:----------:| --------:|-----------|----|:-----------:|-------|
| CPU 1           | numpy         | StarLightCurves  |  **503.686** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with full broadcasting to 3D array|
| CPU 1           | numpy         | StarLightCurves  |  **206.529** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with full broadcasting to 3D array|
| CPU 1           | numpy         | StarLightCurves  |  **563.883** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with full broadcasting to 3D array|
| CPU 1           | numpy         | StarLightCurves  |  **116.558** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with full broadcasting to 3D array|
| CPU 1           | numpy         | StarLightCurves  |  **53.658**  |float64  | 3 |commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1           | numpy         | StarLightCurves  |  **129.130**  |float64  | 3 |commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1           | numpy         | StarLightCurves  |  **72.424**  |float64  | 3 |commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1          | pytorch       | StarLightCurves  |  **248.752** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1          | pytorch       | StarLightCurves  |  **108.819** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1          | pytorch       | StarLightCurves  |  **55.395** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1          | pytorch       | StarLightCurves  |  **141.321** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| CPU 1          | pytorch       | StarLightCurves  |  **82.864** | float64 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with iterative assignment of time-series to clusters |
| GPU           | pytorch       | StarLightCurves  |   **139.560**  | float32 | 3 |           |
| CPU 2           | numpy         | StarLightCurves  |  **439.684** | float32 | 3  | commit: 52a01fe1206fd98c39eeaed0e7199a80d01421b2 with full broadcasting to 3D array|

## Relevant Articles

### Original paper

```plain
Paparrizos J and Gravano L (2015).
k-Shape: Efficient and Accurate Clustering of Time Series.
In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data, series SIGMOD '15,
pp. 1855-1870. ISBN 978-1-4503-2758-9, http://doi.org/10.1145/2723372.2737793. '
```

### Our paper where we used the python implementation
```bibtex
@article{sieve-middleware-2017,
  author       = {J{\"o}rg Thalheim, Antonio Rodrigues, Istemi Ekin Akkus, Pramod Bhatotia, Ruichuan Chen, Bimal Viswanath, Lei Jiao, Christof Fetzer},
  title        = {Sieve: Actionable Insights from Monitored Metrics in Distributed Systems}
  booktitle    = {Proceedings of Middleware Conference (Middleware)},
  year         = {2017},
}
```

### GPU citation
```bibtex
@inproceedings{pykshape,
  title={PyKShape - python implementation of KShape for CPU and GPU},
  author={Dziedzic, Adam and Thalheim, J{\"o}rg and Paparrizos, John},
  booktitle={SIGMOD},
  year={2018}
}
```
