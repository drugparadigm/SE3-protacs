import torch
import pandas as pd
import os
from tqdm import tqdm
from utils import ESMEmbedder

def precompute_embeddings(csv_file, output_dir, device='cuda:2'):
    """
    Precompute ESM embeddings for all protein sequences in the dataset.
    
    Args:
        csv_file: Path to the CSV file containing the data
        output_dir: Directory to save the precomputed embeddings
        device: Device to run ESM model on ('cuda' or 'cpu')
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} samples from {csv_file}")
    
    # Initialize ESM embedder
    print(f"Initializing ESM embedder on {device}...")
    esm_embedder = ESMEmbedder(device=device)
    
    # Track unique sequences to avoid redundant computation
    unique_target_sequences = {}
    unique_e3_sequences = {}
    
    print("Precomputing embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        compound_id = row['compound id']
        target_sequence = row['target_sequence']
        e3_ligase_sequence = row['e3_ligase_sequence']
        
        # Process target sequence
        if target_sequence not in unique_target_sequences:
            target_embedding = esm_embedder.embed_sequence(target_sequence)
            unique_target_sequences[target_sequence] = target_embedding
        else:
            target_embedding = unique_target_sequences[target_sequence]
        
        # Process E3 ligase sequence
        if e3_ligase_sequence not in unique_e3_sequences:
            e3_embedding = esm_embedder.embed_sequence(e3_ligase_sequence)
            unique_e3_sequences[e3_ligase_sequence] = e3_embedding
        else:
            e3_embedding = unique_e3_sequences[e3_ligase_sequence]
        
        # Save embeddings with compound_id
        target_path = os.path.join(output_dir, f'target_{compound_id}.pt')
        e3_path = os.path.join(output_dir, f'e3_ligase_{compound_id}.pt')
        
        torch.save(target_embedding.cpu(), target_path)
        torch.save(e3_embedding.cpu(), e3_path)
    
    print(f"\nPrecomputation complete!")
    print(f"Unique target sequences: {len(unique_target_sequences)}")
    print(f"Unique E3 ligase sequences: {len(unique_e3_sequences)}")
    print(f"Embeddings saved to: {output_dir}")

if __name__ == "__main__":
    # Configure these paths according to your setup
    CSV_FILE = 'data/1979_samples.csv'
    OUTPUT_DIR = 'data/1979_esm3/'
    DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    precompute_embeddings(CSV_FILE, OUTPUT_DIR, DEVICE)