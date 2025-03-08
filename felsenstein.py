import numpy as np
import re
import tempfile
from itertools import count
from Bio import AlignIO, Phylo
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

def get_transition_matrix(t, alpha=1.0):
    """
    Returns the 4x4 transition probability matrix under the Jukes-Cantor model for branch length t.
    """
    exp_factor = np.exp(-4 * alpha * t / 3)
    p_same = 1/4 + 3/4 * exp_factor
    p_diff = 1/4 - 1/4 * exp_factor
    P = np.full((4, 4), p_diff)
    np.fill_diagonal(P, p_same)
    return P

def parse_alignment_file(alignment_file):
    """
    Parses the Nexus alignment file.
    Returns:
      - leaf_data: dict mapping taxon names to an array of shape (n_sites, 4) with one-hot encoding.
      - n_sites: number of sites in the alignment.
    """
    alignment = AlignIO.read(alignment_file, "nexus")
    n_sites = alignment.get_alignment_length()
    # Map nucleotides to one-hot vectors.
    nuc_to_onehot = {
        'A': np.array([1, 0, 0, 0]),
        'C': np.array([0, 1, 0, 0]),
        'G': np.array([0, 0, 1, 0]),
        'T': np.array([0, 0, 0, 1]),
        '-': np.array([0.25, 0.25, 0.25, 0.25]),  # for gaps or ambiguous symbols
        'N': np.array([0.25, 0.25, 0.25, 0.25])
    }
    leaf_data = {}
    for record in alignment:
        seq_array = np.array([nuc_to_onehot.get(nuc.upper(), np.array([0.25, 0.25, 0.25, 0.25]))
                              for nuc in record.seq])
        leaf_data[record.id] = seq_array  # shape: (n_sites, 4)
    return leaf_data, n_sites

def felsenstein_pruning_multi(node, leaf_distributions):
    """
    Recursively computes the likelihood matrix for each node across all sites.
    Each leaf's likelihood is a matrix of shape (n_sites, 4).
    For an internal node, for each child, we compute:
       L_parent(site, i) = product_over_children ( sum_j [ P_ij(child) * L_child(site, j) ] )
    """
    if node.is_terminal():
        if node.name not in leaf_distributions:
            raise ValueError(f"Missing nucleotide data for leaf '{node.name}'")
        node.likelihood = leaf_distributions[node.name]  # shape: (n_sites, 4)
        return node.likelihood
    
    child_likelihoods = []
    for child in node.clades:
        child_lh = felsenstein_pruning_multi(child, leaf_distributions)  # shape: (n_sites, 4)
        t = child.branch_length if child.branch_length is not None else 0.0
        P = get_transition_matrix(t)  # shape: (4,4)
        # For each site, parent's likelihood[i] = sum_j P[i,j] * L_child(site, j)
        child_transformed = np.einsum("ij,sj->si", P, child_lh)  # shape: (n_sites,4)
        child_likelihoods.append(child_transformed)
    
    # Multiply contributions from all children element-wise (per site, per state)
    combined = child_likelihoods[0]
    for child_contrib in child_likelihoods[1:]:
        combined = combined * child_contrib
    node.likelihood = combined
    return node.likelihood

def normalize_likelihoods(node):
    """
    Normalizes the likelihood matrix at each node per site so that the probabilities sum to 1.
    """
    total = np.sum(node.likelihood, axis=1, keepdims=True)  # shape: (n_sites,1)
    # Avoid division by zero
    node.normalized = np.where(total > 0, node.likelihood / total, node.likelihood)
    for child in node.clades:
        normalize_likelihoods(child)

# For converting indices to nucleotides.
index_to_nuc = {0: "A", 1: "C", 2: "G", 3: "T"}

def get_most_likely_nucleotide_string(node):
    """
    Given a node with a normalized likelihood matrix of shape (n_sites, 4),
    returns a string of the most likely nucleotide at each site.
    """
    argmax_indices = np.argmax(node.normalized, axis=1)
    nucs = [index_to_nuc[i] for i in argmax_indices]
    return "".join(nucs)

def assign_internal_names(node, counter):
    """
    Recursively assigns names to internal nodes if they don't have one.
    """
    if not node.is_terminal() and (node.name is None or node.name.strip() == ""):
        node.name = f"Internal_{next(counter)}"
    for child in node.clades:
        assign_internal_names(child, counter)

def extract_all_node_states(node, node_dict):
    """
    Recursively extracts the most likely nucleotide string for each node and
    stores it in node_dict with the node's name as the key.
    """
    if hasattr(node, "normalized"):
        node_dict[node.name] = get_most_likely_nucleotide_string(node)
    else:
        node_dict[node.name] = None
    for child in node.clades:
        extract_all_node_states(child, node_dict)
    return node_dict

def main():
    # Set paths to your files (adjust these paths as needed)
    tree_input_file = "/Users/antoniomoretti/Documents/MrBayes/examples/primates.nex.con.tre"
    alignment_file = "/Users/antoniomoretti/Documents/MrBayes/examples/primates.nex"
    
    # Clean the tree file to remove bracketed metadata
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".nex") as tmp:
        cleaned_tree_file = tmp.name
    clean_nexus_file(tree_input_file, cleaned_tree_file)
    
    # Read the cleaned tree using Biopython's Phylo
    tree = Phylo.read(cleaned_tree_file, "nexus")
    
    # Parse the alignment file to get nucleotide data for each leaf
    leaf_data, n_sites = parse_alignment_file(alignment_file)
    print(f"Alignment has {n_sites} sites.")
    
    # Run Felsenstein's pruning algorithm for all sites (computes likelihood matrix for each node)
    felsenstein_pruning_multi(tree.clade, leaf_data)
    normalize_likelihoods(tree.clade)
    
    # Assign names to internal nodes if they are unnamed.
    internal_name_counter = count(1)
    assign_internal_names(tree.clade, internal_name_counter)
    
    # Extract all node states (most likely nucleotide string per site) into a dictionary.
    node_state_dict = extract_all_node_states(tree.clade, {})
    
    # Print the dictionary (it should have 2N-1 keys for N leaves)
    pprint(node_state_dict)
    
    # Optionally, print the ASCII representation of the tree
    print("\nASCII representation of the tree:")
    Phylo.draw_ascii(tree)

if __name__ == "__main__":
    main()
