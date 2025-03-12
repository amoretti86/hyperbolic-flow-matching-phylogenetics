import json
import re
import tempfile
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import geoopt
from itertools import count
from Bio import Phylo
import os
import matplotlib.pyplot as plt
from datetime import datetime

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
def embed_to_poincare(u):
    """
    Maps u in R^2 to z in the open unit disk via:
        z = u / (1 + ||u||)
    This guarantees ||z|| < 1 for any u.
    """
    norm = torch.norm(u, dim=1, keepdim=True)
    return u / (1 + norm)

def save_embedding_plot(embeddings, node_names, results_folder, epoch):
    """
    Plots current embeddings on the Poincaré disk and saves the figure.
    embeddings: numpy array of shape (n, 2)
    node_names: list of node names corresponding to each embedding.
    results_folder: path to folder for saving figures.
    epoch: current epoch (iteration) number.
    """
    plt.figure(figsize=(8,8))
    # Plot unit circle
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(np.cos(theta), np.sin(theta), 'k-', lw=2, label="Unit Circle")
    
    # Plot each embedding
    for i, name in enumerate(node_names):
        x, y = embeddings[i]
        # Use red for internal nodes (if name contains "Internal" or "Root") else blue.
        if "Internal" in name or name == "Root":
            color = "red"
        else:
            color = "blue"
        plt.scatter(x, y, c=color, s=50)
        plt.text(x + 0.02, y + 0.02, name, fontsize=8, color=color)
    
    plt.title(f"Poincaré Embeddings at Iteration {epoch}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    filename = os.path.join(results_folder, f"epoch_{epoch:04d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def train_poincare_embeddings_geoopt(node_state_dict, parent_child_pairs, num_iters=1000, lr=0.01, margin=1.0, save_interval=100, results_folder=None):
    """
    Trains Poincaré embeddings using geoopt for Riemannian optimization.
    Saves a plot of the current embeddings every `save_interval` iterations.
    
    node_state_dict: dict mapping node names to sequences (for node set).
    parent_child_pairs: list of (parent_name, child_name) tuples.
    num_iters: number of training iterations.
    lr: learning rate.
    margin: margin for the ranking loss.
    save_interval: save a figure every this many iterations.
    results_folder: folder to save results (if None, a new folder with timestamp is created).
    
    Returns a dictionary mapping node names to their final 2D embeddings.
    """
    # Build list of node names.
    node_names = list(node_state_dict.keys())
    idx2name = {i: name for i, name in enumerate(node_names)}
    name2idx = {name: i for i, name in enumerate(node_names)}
    n = len(node_names)
    
    # Define the Poincaré ball manifold from geoopt (curvature c=1).
    manifold = geoopt.PoincareBall(c=1.0)
    
    # Initialize embeddings as manifold parameters.
    u = geoopt.ManifoldParameter(torch.randn(n, 2), manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([u], lr=lr)
    
    # Convert parent-child pairs to indices.
    pos_pairs = [(name2idx[p], name2idx[c]) for (p, c) in parent_child_pairs if p in name2idx and c in name2idx]
    
    best_loss = float('inf')
    best_state = None
    
    # Create results folder if not provided.
    if results_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"results\{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    
    # Save metadata.
    metadata = {
        "num_iters": num_iters,
        "learning_rate": lr,
        "margin": margin,
        "save_interval": save_interval,
        "results_folder": results_folder
    }
    with open(os.path.join(results_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    for it in range(num_iters):
        optimizer.zero_grad()
        emb = u  # u is already on the manifold.
        loss = 0.0
        for i, j in pos_pairs:
            d_pos = manifold.dist(emb[i], emb[j])
            loss += d_pos  # Encourage small positive distance.
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
        
        if it % save_interval == 0:
            current_emb = u.detach().cpu().numpy()
            save_embedding_plot(current_emb, node_names, results_folder, it)
            print(f"Iteration {it}, Loss: {loss.item():.4f}")
            if torch.isnan(loss):
                print("Loss is NaN; stopping training.")
                break
    
    final_embeddings = embed_to_poincare(best_state).detach().cpu().numpy()  # shape: (n,2)
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
    def get_parent_child_pairs(tree):
        pairs = []
        def traverse(node):
            for child in node.clades:
                pairs.append((node.name, child.name))
                traverse(child)
        traverse(tree.clade)
        return pairs
    pairs = get_parent_child_pairs(tree)
    print("Extracted parent-child pairs:")
    print(pairs)
    
    # Train using geoopt.
    new_embedding_dict = train_poincare_embeddings_geoopt(
        node_state_dict,
        pairs,
        num_iters=5000,
        lr=0.008,
        margin=1.0,
        save_interval=100
    )
    
    # Save the embeddings to a JSON file.
    with open("poincare_embeddings_geoopt.json", "w") as f:
        json.dump(new_embedding_dict, f, indent=2)
    print("Poincaré embeddings saved to 'poincare_embeddings_geoopt.json'.")
    
    # Optionally print the embeddings.
    from pprint import pprint
    pprint(new_embedding_dict)

if __name__ == "__main__":
    main()
