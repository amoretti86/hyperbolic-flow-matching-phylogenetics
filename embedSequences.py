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

def embed_to_poincare(u):
    """
    Maps u in R^2 to z in the open unit disk via:
        z = u / (1 + ||u||)
    This guarantees ||z|| < 1 for any u.
    """
    norm = torch.norm(u, dim=1, keepdim=True)
    return u / (1 + norm)

def main():
    # ========================
    # 1. LOAD DATA
    # ========================
    with open("node_state_dict.json", "r") as f:
        node_state_dict = json.load(f)
        print("node state dict:\n", node_state_dict)
    
    # Convert node_state_dict to a list of (name, seq)
    nodes = list(node_state_dict.items())
    node_names = [name for name, seq in nodes]
    sequences = [seq for name, seq in nodes]
    n = len(sequences)
    print(f"Loaded {n} sequences.")
    
    # Build the Hamming distance matrix
    hamming_mat = compute_hamming_matrix(sequences)
    hamming_mat = torch.tensor(hamming_mat, dtype=torch.float32)
    
    # ========================
    # 2. SELECT ROOT NODE
    # ========================
    # Example: Suppose you decided "Internal_10" is the root. 
    # Or if your tree is truly rooted, use that root name.
    root_name = "Internal_10"
    # Find index of root in node_names (or handle if not found)
    if root_name not in node_names:
        raise ValueError(f"Root node '{root_name}' not found in node_state_dict!")
    root_idx = node_names.index(root_name)
    
    # ========================
    # 3. INITIALIZE PARAMETERS
    # ========================
    # We'll embed each node in R^2 as 'u_i', then map to z_i in the Poincaré disk.
    u = torch.randn((n, 2), requires_grad=True)
    optimizer = optim.Adam([u], lr=0.01)
    
    # Strength of the root penalty
    lambda_root = 10.0  # tune this hyperparameter as desired
    
    # ========================
    # 4. TRAINING LOOP
    # ========================
    n_iters = 1000
    for step in range(n_iters):
        optimizer.zero_grad()
        
        # Map to Poincaré disk
        embeddings = embed_to_poincare(u)  # (n,2)
        
        # Compute pairwise hyperbolic distances
        hyper_dist = compute_all_hyperbolic_distances(embeddings)  # (n, n)
        
        # Main distance matching loss
        loss_dist = torch.sum((hyper_dist - hamming_mat)**2)
        
        # Root penalty: encourage root to be near origin
        # Norm of the root node's embedding
        root_embedding = embeddings[root_idx]
        root_norm_sq = torch.sum(root_embedding**2)
        loss_root = lambda_root * root_norm_sq
        
        # Total loss
        loss = loss_dist + loss_root
        
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Iteration {step}, Loss: {loss.item():.4f} "
                  f"(Dist Loss: {loss_dist.item():.4f}, Root Penalty: {loss_root.item():.4f})")
            
            if torch.isnan(loss):
                print("Loss is NaN. Stopping training.")
                break
    
    # ========================
    # 5. RETRIEVE EMBEDDINGS
    # ========================
    final_embeddings = embed_to_poincare(u).detach().numpy()  # shape: (n,2)
    # Check norms
    norms = np.linalg.norm(final_embeddings, axis=1)
    if np.any(norms >= 1.0):
        print("Warning: Some embeddings are on or outside the Poincaré disk boundary!")
    
    # Build embedding dictionary
    embedding_dict = {
        node_names[i]: final_embeddings[i].tolist() for i in range(n)
    }
    
    # Save to JSON
    with open("node_embeddings.json", "w") as f:
        json.dump(embedding_dict, f, indent=2)
    
    print("\nFinal Embeddings (with root penalty):")
    for name, emb in embedding_dict.items():
        print(f"{name}: {emb}")

if __name__ == "__main__":
    main()
