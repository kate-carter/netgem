#correlates a full set of data at a time (all dimensinoalities for the
# listed concepts
# correlates SUBTYPES NOT FEATURES (for mapping in semantic space)

# pip import packages and then..
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import time
import scipy.spatial.distance as dist
import argparse
from scipy.spatial.distance import pdist, squareform
import logging
import os

# List of concepts to process
concept_name = [
    'banana',
    'bottle',
    'car',
    'chocolate',
    'cookie',
    'grass',
    'key',
    'knife',
    'paper',
    'phone',
    'pickle',
    'pillow',
    'pumpkin',
    'table',
    'wood'
]

# Base paths
input_dir = '/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/run_5/embeddings-2/'
output_dir = '/Users/Kate/Documents/Binghamton-SPiNS/wordgen/exploratory_gemini_1536/mahalanobis/'

start_time = time.time()
# set up argument parser for method
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='cosine',
                    help='Method to use for correlation: pearson, spearman, mahalanobis')
args = parser.parse_args()
method = args.method

results = []
#set up target dimensionality so that it processes all of them
target_dimensionality = (3072, 1536, 768, 512, 256)
correlation_by_dim = {dim: {} for dim in target_dimensionality}
dim = target_dimensionality[0]

#construct filename
concept = concept_name[0]
filename = f'{concept}_{dim}.csv'

#loop to process each concept
for concept in concept_name:
    for dim in target_dimensionality:
        file_start = time.time()
        filename = f'{concept}_{dim}.csv'
        input_path = f'{input_dir}{filename}'
        print(f"Processing {concept} at dimension {dim}")
        try:
            # Load data and perform correlation (row×row to create 250×250 matrix)
            matrix_emb = pd.read_csv(f'{input_dir}{filename}', skiprows=8, index_col=0)
            print(f"  Loaded shape: {matrix_emb.shape}")
            X = matrix_emb.values.T  # transpose so each column (word) becomes a row vector for pdist

            if method == 'pearson':
                # Create correlation matrix using pearson correlation
                matrix_corr = matrix_emb.corr(
                    method='pearson'
                ).values
                print(f"  Correlation matrix shape: {matrix_corr.shape}")
            elif method == 'spearman':
                # Create correlation matrix using spearman rank correlation
                matrix_corr = matrix_emb.corr(
                    method='spearman'
                ).values
                print(f"  Correlation matrix shape: {matrix_corr.shape}")
            elif method == 'mahalanobis':
                #Create distance matrix using mahalanobis distance
                VI = np.linalg.pinv(np.cov(X, rowvar=False))
                dists = pdist(X, metric='mahalanobis', VI=VI)
                d = squareform(dists)
                off_diag_distances = d[np.triu_indices_from(d, k=1)]
                
                #convert to similarity using GAUSSIAN kernel
                sigma = np.median(off_diag_distances)
                if sigma > 0:
                    sim01 = np.exp(-(d ** 2) / (2 * sigma ** 2))   # [0,1]
                else:
                    sim01 = np.ones_like(d)
                matrix_corr = sim01


                # exponential kernel similarity conversion (Euclidean/Laplacian form)
                #sigma = np.median(off_diag_distances)
                #sigma = max(sigma, 1e-12)  # guard
                #matrix_corr = np.exp(-d / sigma)
                print(f"  Mahalanobis distance matrix shape: {matrix_corr.shape}")
            elif method == "cosine":
                # create correlation matrix using cosine similarity
                dists = pdist(X, metric='cosine')
                matrix_corr = 1 - squareform(dists)
                print(f"  Cosine similarity matrix shape: {matrix_corr.shape}")

            # Min/max from off-diagonal only (diagonal must not set the scale)
            iu = np.triu_indices_from(matrix_corr, k=1)
            off_diag = matrix_corr[iu]
            min_val = np.min(off_diag)
            max_val = np.max(off_diag)
            print(f'min/max before normalization: {min_val}, {max_val}')

            # Normalize to [-1, 1] (full matrix; diagonal set after)
            if method == 'mahalanobis':
                if max_val > min_val:
                    matrix_corr = 2 * ((matrix_corr - min_val) / (max_val - min_val)) - 1
                else:
                    matrix_corr = np.zeros_like(matrix_corr, dtype=float)
                iu = np.triu_indices_from(matrix_corr, k=1)
                off_diag2 = matrix_corr[iu]
                min_val2 = np.min(off_diag2)
                max_val2 = np.max(off_diag2)
                print(f'min/max after normalization: {min_val2}, {max_val2}')
            else:
                # norm to [-1, 1]
                if max_val > min_val:
                    matrix_corr = 2 * ((matrix_corr - min_val) / (max_val - min_val)) - 1
                else:
                    matrix_corr = np.zeros_like(matrix_corr, dtype=float)
                iu = np.triu_indices_from(matrix_corr, k=1)
                off_diag2 = matrix_corr[iu]
                min_val2 = np.min(off_diag2)
                max_val2 = np.max(off_diag2)
                print(f'min/max after normalization: {min_val2}, {max_val2}'
                                )

            # Convert diagonal from 1's to 0's
            matrix_corr = matrix_corr.copy()
            np.fill_diagonal(matrix_corr, 0)
            
            # Save matrix with consistent naming: concept_corr.csv
            output_file = f'{output_dir}{concept}_{dim}_{method}.csv'
            pd.DataFrame(matrix_corr).to_csv(output_file, header=False, index=False)

            elapsed = time.time() - file_start
            print(f"  ✓ Saved: {concept}_{dim}_{method}.csv ({elapsed:.2f}s)")
            results.append({
                'concept': concept,
                'method': method,
                'matrix_shape': matrix_corr.shape
            })
        except Exception as e:
            logging.error(f"ERROR processing {concept}: {e}")
            continue

total_time = time.time() - start_time
print("\n" + "=" * 60)
print(f"Completed! Processed {len(concept_name)} concepts with {target_dimensionality} dimensionality elapsed time of {total_time / 60:.2f} minutes")
print(f"Output directory: {output_dir}")
