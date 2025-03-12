import json
import re
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from pprint import pprint

def clean_nexus_file(input_file, output_file):
    """
    Reads a Nexus file and removes extra bracketed metadata starting with "[&" 
    and ending with "]", then writes the cleaned content to output_file.
    """
    with open(input_file, 'r') as f:
        contents = f.read()
    # Remove any text starting with [& and ending with ]
    cleaned = re.sub(r'\[&[^]]*\]', '', contents)
    with open(output_file, 'w') as f:
        f.write(cleaned)

def load_tree(tree_input_file):
    """
    Cleans and reads the Nexus tree file using Biopython.
    """
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".nex") as tmp:
        cleaned_tree_file = tmp.name
    clean_nexus_file(tree_input_file, cleaned_tree_file)
    tree = Phylo.read(cleaned_tree_file, "nexus")
    return tree

def assign_internal_names(node, counter):
    """
    Recursively assigns names to internal nodes if they don't have one.
    Uses counter (e.g., itertools.count) to generate unique names.
    """
    if not node.is_terminal() and (node.name is None or node.name.strip() == ""):
        node.name = f"Internal_{next(counter)}"
    for child in node.clades:
        assign_internal_names(child, counter)

def plot_poincare(embedding_dict, tree):
    """
    Plots the Poincaré disk (unit circle), overlays the node embeddings from embedding_dict,
    and draws the tree structure by connecting parent and child nodes.
    
    Leaf nodes are colored blue (with labels); internal nodes are colored red.
    """
    # Create unit circle for Poincaré disk
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(circle_x, circle_y, 'k-', lw=1)  # unit circle boundary

    # Recursive function to draw edges from parent to children
    def plot_edges(node):
        parent_name = node.name
        if parent_name in embedding_dict:
            parent_coord = embedding_dict[parent_name]
            for child in node.clades:
                child_name = child.name
                if child_name in embedding_dict:
                    child_coord = embedding_dict[child_name]
                    plt.plot([parent_coord[0], child_coord[0]],
                             [parent_coord[1], child_coord[1]], 'gray', lw=0.5)
                plot_edges(child)
    
    plot_edges(tree.clade)
    
    # Plot nodes: blue for leaves (with labels), red for internal nodes.
    # To avoid duplicate legend entries, we collect handles.
    leaf_plotted = False
    internal_plotted = False
    for node in tree.find_clades():
        if node.name in embedding_dict:
            coord = embedding_dict[node.name]
            x, y = coord[0], coord[1]
            
            if node.is_terminal():
                # Plot leaf node in blue
                if not leaf_plotted:
                    plt.scatter(x, y, c='blue', s=50, label='Leaf')
                    leaf_plotted = True
                else:
                    plt.scatter(x, y, c='blue', s=50)
                
                # Add text label (slightly offset to avoid overlap)
                plt.text(x + 0.02, y + 0.02, node.name,
                         fontsize=8, color='blue', 
                         ha='left', va='bottom')
            else:
                # Plot internal node in red
                if not internal_plotted:
                    plt.scatter(x, y, c='red', s=50, label='Internal')
                    internal_plotted = True
                else:
                    plt.scatter(x, y, c='red', s=50)
    
    plt.title("Poincaré Disk Embedding of Tree Nodes")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.legend()
    plt.savefig("Embedding.pdf", bbox_inches='tight')
    plt.show()


def main():
    # Set paths to your files (adjust these paths as needed)
    tree_input_file = "/Users/antoniomoretti/Documents/MrBayes/examples/primates.nex.con.tre"
    embeddings_file = "node_embeddings.json"  # file containing node embeddings from previous analysis
    
    # Load the tree (with cleaned metadata)
    tree = load_tree(tree_input_file)
    
    # Assign names to internal nodes if missing
    from itertools import count
    internal_name_counter = count(1)
    assign_internal_names(tree.clade, internal_name_counter)
    
    # Load node embeddings from JSON
    with open(embeddings_file, "r") as f:
        embedding_dict = json.load(f)
    
    # Optionally, print the embedding dictionary to verify
    pprint(embedding_dict)
    
    # Plot the embeddings on the Poincaré disk with the tree structure
    plot_poincare(embedding_dict, tree)

if __name__ == "__main__":
    main()
