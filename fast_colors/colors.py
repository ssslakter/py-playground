# AUTOGENERATED! DO NOT EDIT! File to edit: colors.ipynb.

# %% auto 0
__all__ = ['extract_colors', 'scatter_plotly', 'compress_img', 'k_means', 'segment_k_means']

# %% colors.ipynb 1
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from fastcore.all import *
from PIL import Image

# %% colors.ipynb 4
def extract_colors(img):
    arr = np.array(img)/255 # normalize to unit cube
    cols  = arr.reshape(-1, 3)
    return cols.T

# %% colors.ipynb 6
def scatter_plotly(cols):
    x,y,z = cols
    return go.Figure(data=[go.Scatter3d(x=x,y=y,z=z, mode='markers', marker=dict(size=2, color=cols.T))])

def compress_img(img, scale=256):
    w, h = img.size
    return img.resize((min(scale,w), min(scale,h)))

# %% colors.ipynb 11
import torch

def k_means(points, k=5, eps=1e-4):
    centers = points.unique(dim=0)
    centroids = centers[torch.randperm(centers.size(0))[:k]]
    
    while True:
        clusters = torch.argmin(torch.cdist(points, centroids), dim=1)
        new_centroids = torch.stack([points[clusters == i].mean(dim=0) for i in range(k)])
        if torch.all(torch.abs(new_centroids - centroids) < eps):
            break
        centroids = new_centroids
    
    return clusters, centroids

def segment_k_means(img, k=5, eps=1e-2):
    points = img.reshape(-1, img.shape[-1])
    clusters, centroids = k_means(points, k, eps)
    points = centroids[clusters].reshape(img.shape)
    return Image.fromarray((points*255).type(torch.uint8).cpu().numpy())
