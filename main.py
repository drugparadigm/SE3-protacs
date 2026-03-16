import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


from dataset import PROTACDataset, collater
from model import *
from train import train, valids

BATCH_SIZE = 2
EPOCH = 100
LEARNING_RATE = 0.0001
TRAIN_NAME = "SE3-PROTACs_1979_ESM3"
root = "data"
logging.basicConfig(filename="log/" + TRAIN_NAME + ".log", filemode="w", level=logging.DEBUG)
RANDOM_SEED = 42
torch.cuda.manual_seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    set_seed(RANDOM_SEED)
    # ==================== DEFINE PATHS ====================
    train_csv = 'data/train.csv'
    val_csv = 'data/val.csv'
    
    train_mol2_dir = 'data/mol2_files/'
    val_mol2_dir = 'data/mol2_files/'
    
    # Separate embedding directories for train and val
    train_embeddings_dir = 'data/esm/'
    val_embeddings_dir = 'data/esm/'
    
    # ==================== CHECK EMBEDDINGS EXIST ====================
    if not os.path.exists(train_embeddings_dir):
        print("\n" + "="*80)
        print("❌ ERROR: Train embeddings directory not found!")
        print("="*80)
        print(f"Expected location: {train_embeddings_dir}")
        logging.error(f"Train embeddings directory not found: {train_embeddings_dir}")
        return
    
    if not os.path.exists(val_embeddings_dir):
        print("\n" + "="*80)
        print("❌ ERROR: Validation embeddings directory not found!")
        print("="*80)
        print(f"Expected location: {val_embeddings_dir}")
        logging.error(f"Validation embeddings directory not found: {val_embeddings_dir}")
        return
    
    # Count embedding files for verification
    train_embedding_files = [f for f in os.listdir(train_embeddings_dir) if f.endswith('.pt')]
    val_embedding_files = [f for f in os.listdir(val_embeddings_dir) if f.endswith('.pt')]
    
    print(f"\n✅ Found {len(train_embedding_files)} embedding files in {train_embeddings_dir}")
    print(f"✅ Found {len(val_embedding_files)} embedding files in {val_embeddings_dir}")
    
    logging.info(f"Found {len(train_embedding_files)} training embedding files")
    logging.info(f"Found {len(val_embedding_files)} validation embedding files")
    
    # ==================== LOAD DATASETS SEPARATELY ====================
    print("\nLoading training dataset...")
    try:
        train_dataset = PROTACDataset(train_mol2_dir, train_csv, train_embeddings_dir)
        print(f"✅ Training dataset loaded successfully: {len(train_dataset)} samples")
        logging.info(f"Training dataset loaded successfully: {len(train_dataset)} samples")
    except Exception as e:
        print(f"\n❌ Error loading training dataset: {e}")
        logging.error(f"Error loading training dataset: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    print("\nLoading validation dataset...")
    try:
        val_dataset = PROTACDataset(val_mol2_dir, val_csv, val_embeddings_dir)
        print(f"✅ Validation dataset loaded successfully: {len(val_dataset)} samples")
        logging.info(f"Validation dataset loaded successfully: {len(val_dataset)} samples")
    except Exception as e:
        print(f"\n❌ Error loading validation dataset: {e}")
        logging.error(f"Error loading validation dataset: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
    logging.info(f"Training data: {len(train_dataset)}")
    logging.info(f"Validation data: {len(val_dataset)}")
    
    # ==================== CREATE DATALOADERS ====================
    num_workers = 2
    
    print(f"\nCreating data loaders...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Num workers: {num_workers}")
    
    trainloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collater,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logging.info(f"DataLoaders created with batch_size={BATCH_SIZE}, num_workers={num_workers}")
    logging.info(f"Training batches: {len(trainloader)}")
    logging.info(f"Validation batches: {len(valloader)}")

    # ==================== INITIALIZE MODELS ====================
    print("\nInitializing models...")
    
    ligase_model = ESMWrapper()
    target_model = ESMWrapper()
    ESM_DIM=128
    
    target_ligand_model = GraphTransformer(num_embeddings=10,dim=ESM_DIM)
    ligase_ligand_model = GraphTransformer(num_embeddings=10,dim=ESM_DIM)
    linker_model = GraphTransformer(num_embeddings=10,dim=ESM_DIM)
    
    model = Model(
        ligase_ligand_model,
        ligase_model,
        target_ligand_model,
        target_model,
        linker_model,
        dim=960
    )

    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    logging.info(f"Model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    # ==================== SETUP DEVICE & TENSORBOARD ====================
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    logging.info(f"Using device: {device}")
    
    writer = SummaryWriter(f'runs/{TRAIN_NAME}')
    print(f"TensorBoard logging to: runs/{TRAIN_NAME}")
    print(f"  View with: tensorboard --logdir=runs")
    logging.info(f"TensorBoard writer initialized: runs/{TRAIN_NAME}")
    model= model.to(device)

    # ==================== TRAINING ====================
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    logging.info("="*80)
    logging.info("STARTING TRAINING")
    logging.info("="*80)
    
    # Train with proper validation set
    model = train(
        model,
        train_loader=trainloader,
        valid_loader=valloader,
        device=device,
        writer=writer,
        LOSS_NAME=TRAIN_NAME,
        batch_size=BATCH_SIZE,
        epoch=EPOCH,
        lr=LEARNING_RATE,
        accumulation_steps=8,
    )
    
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    logging.info("✅ TRAINING COMPLETE")
    
    writer.close()

if __name__ == "__main__":
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()