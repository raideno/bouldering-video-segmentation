import math
import torch

import matplotlib.pyplot as plt

def generate_positional_encoding(position: int, dimension: int):
    pe = torch.zeros(dimension)
    
    for i in range(0, dimension, 2):
        w_k = 1 / (10000 ** (i / dimension))
        pe[i] = math.sin(w_k * position)
        if i + 1 < dimension:
            pe[i + 1] = math.cos(w_k * position)
            
    return pe

def plot_positional_encoding(dimension:int=128, max_position:int=100):
    positional_encodings = torch.stack([generate_positional_encoding(position, dimension) for position in range(max_position)])

    plt.figure(figsize=(12, 6))
    plt.imshow(positional_encodings, cmap='bwr', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Positional Encoding Visualization')
    plt.xlabel('Depth (d_model)')
    plt.ylabel('Position')
    plt.show()