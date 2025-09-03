import os
import logging
from pathlib import Path

import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import PROTACDataset, collater
from model import *
from train import train

BATCH_SIZE = 1
EPOCH = 100
TRAIN_RATE = 0.8
LEARNING_RATE = 0.0001
TRAIN_NAME = "SE3-PROTACs"
root = "data"
logging.basicConfig(filename="log/" + TRAIN_NAME + ".log", filemode="w", level=logging.DEBUG)
RANDOM_SEED = 42
torch.cuda.manual_seed(RANDOM_SEED)
       
def main():
    print("Loading dataset...")
    protac_set = PROTACDataset('data/mol2_files/',
                               'data/train.csv')
    data_size = len(protac_set)
    train_size = int(data_size * TRAIN_RATE)
    test_size = data_size - train_size
    logging.info(f"all data: {data_size}")
    logging.info(f"train data: {train_size}")
    logging.info(f"test data: {test_size}")
    train_dataset, test_dataset = torch.utils.data.random_split(protac_set, lengths=[train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=collater)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collater)

    ligase_model = ESMWrapper()
    target_model = ESMWrapper()
    
    target_ligand_model = GraphTransformer(num_embeddings=10)
    ligase_ligand_model = GraphTransformer(num_embeddings=10)
    linker_model = GraphTransformer(num_embeddings=10)
    
    model = Model(
        ligase_ligand_model,
        ligase_model,
        target_ligand_model,
        target_model,
        linker_model
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(f'runs/{TRAIN_NAME}')

    model = train(
        model,
        train_loader=trainloader,
        valid_loader=testloader,
        device=device,
        writer=writer,
        LOSS_NAME=TRAIN_NAME,
        batch_size=BATCH_SIZE,
        epoch=EPOCH,
        lr=LEARNING_RATE,
        accumulation_steps=8,
    )

if __name__ == "__main__":
    Path('log').mkdir(exist_ok=True)
    Path('model').mkdir(exist_ok=True)
    main()
