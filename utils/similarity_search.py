import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils # allows utilizing torch tensors as input to faiss indexes and functions

def inverse_distance_kernel(sq_distances, delta = 1e-3):
    """
    Kernel used in Pritzel et. al, 2017
    """
    # https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702
    return 1 / (torch.sqrt(sq_distances + 1e-8) + delta)

def knn_search(queries, data, k):
    """
    Perform exact knn search (should be replaced with approximate) 

    Returns the distance and indices of k nearest neighbours for each query
    """
    # if torch.cuda.is_available():
    #     res = faiss.StandardGpuResources()
    #     # res.setDefaultNullStreamAllDevices()
    #     # res.setTempMemory(64 * 1024 * 1024)
    #     return faiss.knn_gpu(res, data.detach().cpu().numpy(), queries.detach().cpu().numpy(), k) # in version 1.7.0 queries and data are swapped in knn_gpu comapred to knn


    queries, data = queries.detach().cpu().numpy(), data.detach().cpu().numpy()
    return faiss.knn(queries, data, k)

def combine_by_key(keys, values, op = 'max', alpha = None):
    """
    Combines duplicate keys' values using the operator op (max or mean)
    """
    keys = [tuple(key) for key in keys.detach().cpu().numpy()]
    ks, vs = [], []
    key_map = {}

    if alpha is not None:
    # update repeated values in batch using q learning 
        for i, key in enumerate(keys):
            if key in key_map:
                vs[key_map[key]] += alpha * (values[i] - vs[key_map[key]])
            else:
                key_map[key] = len(ks)
                ks.append(key)
                vs.append(values[i]) 
    elif op == 'max':
        for i, key in enumerate(keys):
            if key in key_map:
                idx = key_map[key]
                old_val = vs[idx]
                vs[idx] = values[i] if old_val < values[i] else old_val
            else:
                key_map[key] = len(ks)
                ks.append(key)
                vs.append(values[i])
    elif op == 'mean':
        for i, key in enumerate(keys):
            if key in key_map:
                # update average using stored average, running count, and new value
                idx, n = key_map[key]
                vs[idx] = (vs[idx] * n + values[i]) / (n + 1)
                key_map[key][1] += 1
            else:
                key_map[key] = [len(ks), 1] # store idx in new arrays and running count
                ks.append(key)
                vs.append(values[i])

    return ks, np.float32(vs) # ensure float32 type