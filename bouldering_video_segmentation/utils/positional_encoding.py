import math
import torch

def generate_positional_encoding(position: int, dimension: int):
    pe = torch.zeros(dimension)
    
    for i in range(0, dimension, 2):
        w_k = 1 / (10000 ** (i / dimension))
        pe[i] = math.sin(w_k * position)
        if i + 1 < dimension:
            pe[i + 1] = math.cos(w_k * position)
            
    return pe