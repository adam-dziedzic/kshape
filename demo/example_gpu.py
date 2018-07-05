from kshape.core_gpu import kshape_gpu, zscore_gpu

time_series = [[1,2,3,4,5], [0,1,2,3,4], [3,2,1,0,-1], [1,2,2,3,3]]
cluster_num = 2
clusters = kshape_gpu(zscore_gpu(time_series), cluster_num, device="cuda")
print("centroids and clusters: ", clusters)
second_centroid = clusters[1][0].numpy()
print("second centroid: ", second_centroid)
# can return (there is some randomness involved in the algorithm so this can differ): [(tensor([-1.2511,  1.3528, -0.5106,  0.5652, -0.1564]), [3]), (tensor([-1.3289, -0.8265,  0.7324,  0.6802,  0.7428]), [0, 1, 2])]