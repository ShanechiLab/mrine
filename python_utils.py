import torch
import numpy as np


def carry_to_device(data, device, dtype=torch.float32):
    '''
    Carrys the data into specified device. If data is dictionary, or a list, it recurses until it reaches to np.ndarray or torch.Tensor

    data: torch.Tensor/dict/list/np.ndarray, data to carry to the device
    device: str, device to carry to
    dtype: torch.dtype, dtype to convert np.ndarray 
    '''
    if torch.is_tensor(data):
        return data.to(device)
    
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=dtype).to(device)

    elif isinstance(data, dict):
        for key in data.keys():
            data[key] = carry_to_device(data[key], device)
        return data
    
    elif isinstance(data, list):
        for i, d in enumerate(data):
            data[i] = carry_to_device(d, device)
        return data

    else:
        return data
    

def jitter_inv(matrix, jitter=1e-4):
    '''
    Computes matrix inverse after adding jitter for numerical stability

    matrix: torch.Tensor, matrix to invert
    jitter: float, jitter to add before inversion
    '''
    return torch.linalg.inv(matrix + torch.eye(matrix.shape[-1], device=matrix.device) * jitter)


def extract_dict_from_key(dict_, key_):
    '''
    Flattens a dictionary with 2 nested levels

    dict_: dict, dictionary to flatten
    key_: dict key, key to flatten across
    '''
    if not isinstance(dict_[key_], dict):
        return dict_
    
    dict_inside = dict_[key_]
    for key, val in dict_inside.items():
        dict_[f'{key_}_{key}'] = val
    
    dict_.pop(key_, None)
    return dict_


def convert_to_tensor(x, cat_dim=0):
    '''
    Converts a list, np.ndarray to torch.Tensor

    x:  torch.Tensor, list, or np.ndarray to convert
    cat_dim: int, if x is a list, it is concatenated across cat_dim before converting into a tensor
    '''
    if isinstance(x, np.ndarray): 
        return torch.tensor(x, dtype=torch.float32) # use np.ndarray as middle step so that function works with tf tensors as well
    elif isinstance(x, list):
        return torch.cat(x, dim=cat_dim, dtype=torch.float32) 
    elif isinstance(x, torch.Tensor) and x.dtype == torch.float64: # change dtype to float32
        return x.float()
    else:
        return x


def flatten_dict(dictionary, level = []):
    '''
    Flattens a nested dictionary by creating a master key by putting '.' between nested keys. credit: https://stackoverflow.com/questions/6037503/python-unflatten-dict. 

    dictionary: dict, dictionary to flatten
    '''
    tmp_dict = {}
    for key, val in dictionary.items():
        if isinstance(val, dict):
            tmp_dict.update(flatten_dict(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict
    

def unflatten_dict(dictionary):
    '''
    Unflattens a nested dictionary by splitting master key which is joint string of individual keys. credit: https://stackoverflow.com/questions/6037503/python-unflatten-dict. 

    dictionary: dict, nested dictionary to unflatten
    '''
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict

