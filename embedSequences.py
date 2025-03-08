import json
import torch
import torch.optim as optim
import numpy as np

def hamming_distance(seq1, seq2):
    """Computes the Hamming distance between two equal-length strings."""
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))

def compute_hamming_matrix(sequences):
    """Computes an (n x n) matrix of Hamming distances between a list of sequences."""
    n = len(sequences)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = hamming_distance(sequences[i], sequences[j])
    return matrix

def hyperbolic_distance(u, v, eps=1e-5):
    """
    Computes the hyperbolic distance between two 2D points u and v in the Poincaré disk.
    Uses the formula:
        d(u,v) = arccosh( 1 + (2||u-v||^2)/((1-||u||^2)(1-||v||^2)) )
    Adds eps for numerical stability.
    """
    # Ensure norms are less than 1
    norm_u = torch.clamp(torch.norm(u), max=0.99)
    norm_v = torch.clamp(torch.norm(v), max=0.99)
    diff = u - v
    diff_norm_sq = torch.sum(diff**2)
    # Add eps to denominators to prevent division by zero.
    denom = (1 - norm_u**2 + eps) * (1 - norm_v**2 + eps)
    argument = 1 + (2 * diff_norm_sq) / denom
    # Ensure argument is >= 1 + eps for acosh stability
    argument = torch.clamp(argument, min=1 + eps)
    return torch.acosh(argument)

def compute_all_hyperbolic_distances(embeddings):
    """
    Computes the pairwise hyperbolic distance matrix for embeddings.
    embeddings: tensor of shape (n, 2)
    Returns a tensor of shape (n, n)
    """
    n = embeddings.shape[0]
    dist_matrix = torch.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = hyperbolic_distance(embeddings[i], embeddings[j])
    return dist_matrix

def main():
    # Load node_state_dict from JSON file (each key maps to a nucleotide string)
    with open("node_state_dict.json", "r") as f:
        node_state_dict = json.load(f)
    
    # Create a list of (node_name, sequence)
    nodes = list(node_state_dict.items())
    node_names = [name for name, seq in nodes]
    sequences = [seq for name, seq in nodes]
    n = len(sequences)
    print(f"Loaded {n} sequences.")
    
    # Compute the Hamming distance matrix (as a NumPy array) and then convert to a tensor.
    hamming_mat = compute_hamming_matrix(sequences)
    hamming_mat = torch.tensor(hamming_mat, dtype=torch.float32)
    
    # Initialize learnable parameters u for embeddings in R^2.
    # We use the transformation z = tanh(u) to ensure that the embeddings lie inside the Poincaré disk.
    u = torch.randn((n, 2), requires_grad=True)
    optimizer = optim.Adam([u], lr=0.001)  # use a lower learning rate for stability
    
    # Optimize for a number of iterations.
    n_iters = 1000
    for iter in range(n_iters):
        optimizer.zero_grad()
        embeddings = torch.tanh(u)  # Embedded points in the Poincaré disk, shape: (n,2)
        hyper_dist = compute_all_hyperbolic_distances(embeddings)  # (n, n)
        loss = torch.sum((hyper_dist - hamming_mat)**2)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Iteration {iter}, Loss: {loss.item()}")
            # Check for nan
            if torch.isnan(loss):
                print("Loss is nan, breaking out.")
                break
    
    # Retrieve final embeddings as a NumPy array.
    final_embeddings = torch.tanh(u).detach().numpy()  # shape: (n,2)
    
    # Create a dictionary mapping each node's name to its embedding (a list of two floats)
    embedding_dict = {node_names[i]: final_embeddings[i].tolist() for i in range(n)}
    
    # Save the embeddings to a JSON file.
    with open("node_embeddings.json", "w") as f:
        json.dump(embedding_dict, f, indent=2)
    
    print("\nFinal Embeddings:")
    for name, emb in embedding_dict.items():
        print(f"{name}: {emb}")

if __name__ == "__main__":
    main()
