import json
import re
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import geoopt
from itertools import count
from Bio import Phylo

# --- Utility functions for cleaning and loading the tree ---
def clean_nexus_file(input_file, output_file):
    with open(input_file, 'r') as f:
        contents = f.read()
    cleaned = re.sub(r'\[&[^]]*\]', '', contents)
    with open(output_file, 'w') as f:
        f.write(cleaned)

def load_tree(tree_input_file):
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".nex") as tmp:
        cleaned_tree_file = tmp.name
    clean_nexus_file(tree_input_file, cleaned_tree_file)
    tree = Phylo.read(cleaned_tree_file, "nexus")
    return tree

def assign_internal_names(node, counter):
    if not node.is_terminal() and (node.name is None or node.name.strip() == ""):
        node.name = f"Internal_{next(counter)}"
    for child in node.clades:
        assign_internal_names(child, counter)

def get_parent_child_pairs(tree):
    pairs = []
    def traverse(node):
        for child in node.clades:
            pairs.append((node.name, child.name))
            traverse(child)
    traverse(tree.clade)
    return pairs

# --- Geoopt Poincaré embedding training ---
def train_poincare_embeddings_geoopt(node_state_dict, parent_child_pairs, num_iters=5000, lr=0.008, margin=1.0):
    """
    Trains Poincaré embeddings using geoopt for Riemannian optimization.
    Here, each node is directly optimized on the Poincaré ball.
    
    node_state_dict: dict mapping node names to sequences (for node set).
    parent_child_pairs: list of (parent_name, child_name) tuples.
    """
    # Build list of node names.
    node_names = list(node_state_dict.keys())
    idx2name = {i: name for i, name in enumerate(node_names)}
    name2idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    
    # Define the Poincaré ball manifold from geoopt (curvature c=1).
    manifold = geoopt.PoincareBall(c=1.0)
    
    # Initialize embeddings as manifold parameters. They will remain in the ball.
    # Note: geoopt.ManifoldParameter automatically projects parameters onto the manifold.
    u = geoopt.ManifoldParameter(torch.randn(n, 2), manifold=manifold)
    
    optimizer = geoopt.optim.RiemannianAdam([u], lr=lr)
    
    # Convert parent-child pairs to indices.
    pos_pairs = [(name2idx[p], name2idx[c]) for (p, c) in parent_child_pairs if p in name2idx and c in name2idx]
    
    best_loss = float('inf')
    best_state = None
    
    # Use the manifold's distance function for hyperbolic distances.
    for it in range(num_iters):
        optimizer.zero_grad()
        # u is already on the manifold, so we use it directly.
        emb = u  # shape: (n,2)
        loss = 0.0
        
        # For each positive pair, minimize the hyperbolic distance.
        for i, j in pos_pairs:
            d_pos = manifold.dist(emb[i], emb[j])
            loss += d_pos  # Encourage small distance for positive pairs.
            
            # Negative sampling: choose a random negative index k.
            k = np.random.choice(n)
            if k == i or k == j:
                continue
            d_neg = manifold.dist(emb[i], emb[k])
            loss += F.relu(margin + d_pos - d_neg)
            
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = u.clone().detach()
            
        if it % 100 == 0:
            print(f"Iteration {it}, Loss: {loss.item():.4f}")
            if torch.isnan(loss):
                print("Loss is NaN; stopping training.")
                break
                
    final_embeddings = best_state.cpu().numpy()  # shape: (n,2)
    embedding_dict = {idx2name[i]: final_embeddings[i].tolist() for i in range(n)}
    return embedding_dict

def main():
    # Load node_state_dict from JSON file.
    with open("node_state_dict.json", "r") as f:
        node_state_dict = json.load(f)
    
    # Load the tree and assign internal names.
    tree_input_file = "/Users/antoniomoretti/Documents/MrBayes/examples/primates.nex.con.tre"
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".nex") as tmp:
        cleaned_tree_file = tmp.name
    clean_nexus_file(tree_input_file, cleaned_tree_file)
    tree = Phylo.read(cleaned_tree_file, "nexus")
    assign_internal_names(tree.clade, count(1))
    if tree.root.name is None:
        tree.root.name = "Root"
    
    # Extract parent-child pairs.
    pairs = get_parent_child_pairs(tree)
    print("Extracted parent-child pairs:")
    print(pairs)
    
    # Train using geoopt.
    new_embedding_dict = train_poincare_embeddings_geoopt(node_state_dict, pairs, num_iters=10000, lr=0.005, margin=1.0)
    
    # Save embeddings to JSON.
    with open("poincare_embeddings_geoopt.json", "w") as f:
        json.dump(new_embedding_dict, f, indent=2)
    print("Poincaré embeddings saved to 'poincare_embeddings_geoopt.json'.")
    
    # Optionally print embeddings.
    from pprint import pprint
    pprint(new_embedding_dict)

if __name__ == "__main__":
    main()
