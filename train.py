import logging
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

def valids(model, test_loader, device):
    """Validation function with tqdm progress bar"""
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true, y_pred, y_score = [], [], []
        total_loss, total_samples = 0.0, 0
        
        for data_sample in tqdm(test_loader, desc="Validating", leave=False):
            y = data_sample['label'].to(device)
            batch_size = y.size(0)
            outputs, _, _ = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target'].to(device),
                data_sample['linker'].to(device),
            )
            
            loss = criterion(outputs, y)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            probs = torch.nn.functional.softmax(outputs, 1)
            y_score.extend(probs[:, 1].cpu().tolist())
            y_pred.extend(torch.max(outputs, 1)[1].cpu().tolist())
            y_true.extend(y.cpu().tolist())

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = accuracy_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_score)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        aupr = average_precision_score(y_true, y_score)
        
        model.train()
        return avg_loss, accuracy, auroc, precision, recall, aupr


def train(model, lr=0.0001, epoch=100, train_loader=None, valid_loader=None, 
          device=None, writer=None, LOSS_NAME=None, batch_size=None, 
          accumulation_steps=8, patience=10, min_lr=1e-6):
    
    model = model.to(device)
    best_model_params = None
    best_val_auc = float('-inf')
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False
    
    weight = torch.Tensor([0.8158508, 1.29151292]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, 
                                  verbose=True, min_lr=min_lr)
    
    val_loss, val_acc, auroc, precision, recall, AUPR = valids(model, valid_loader, device)
    logging.info(f'Initial Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, '
                 f'AUROC: {auroc:.4f}, Precision: {precision:.4f}, '
                 f'Recall: {recall:.4f}, AUPR: {AUPR:.4f}')
    
    for epo in range(epoch):
        if early_stop:
            logging.info(f'Early stopping triggered at epoch {epo}')
            break
            
        model.train()
        running_loss, total_num = 0.0, 0
        opt.zero_grad()
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epo+1}/{epoch}")
        
        for i, data_sample in train_bar:
            outputs, _, _ = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target'].to(device),
                data_sample['linker'].to(device),
            )

            y = data_sample['label'].to(device)
            current_batch_size = y.size(0)
            
            loss = criterion(outputs, y)
            loss = loss / accumulation_steps
            loss.backward()
            
            running_loss += loss.item() * accumulation_steps * current_batch_size
            total_num += current_batch_size
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
            
            train_bar.set_postfix({"loss": f"{(running_loss / total_num):.4f}"})
        
        if (i + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
        
        avg_loss = running_loss / total_num if total_num > 0 else 0
        
        val_loss, val_acc, auroc, precision, recall, AUPR = valids(model, valid_loader, device)
        scheduler.step(auroc)
        
        if writer:
            writer.add_scalar(f"{LOSS_NAME}/train_loss", avg_loss, epo)
            writer.add_scalar(f"{LOSS_NAME}/val_loss", val_loss, epo)
            writer.add_scalar(f"{LOSS_NAME}/val_acc", val_acc, epo)
            writer.add_scalar(f"{LOSS_NAME}/auroc", auroc, epo)
            writer.add_scalar(f"{LOSS_NAME}/precision", precision, epo)
            writer.add_scalar(f"{LOSS_NAME}/recall", recall, epo)
            writer.add_scalar(f"{LOSS_NAME}/AUPR", AUPR, epo)
            writer.add_scalar(f"{LOSS_NAME}/learning_rate", opt.param_groups[0]['lr'], epo)
        
        logging.info(f'Epoch {epo:3d} | Train Loss: {avg_loss:.4f} | '
                     f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | '
                     f'AUROC: {auroc:.4f} | Precision: {precision:.4f} | '
                     f'Recall: {recall:.4f} | AUPR: {AUPR:.4f} | '
                     f'LR: {opt.param_groups[0]["lr"]:.2e}')
        
        if auroc > best_val_auc:
            best_val_auc = auroc
            best_model_params = model.state_dict()
            best_epoch = epo
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': best_model_params,
                'epoch': best_epoch,
                'val_auroc': best_val_auc,
                'optimizer_state_dict': opt.state_dict()
            }, f"model/{LOSS_NAME}_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
    
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_params,
        'val_auroc': best_val_auc,
    }, f"model/{LOSS_NAME}_final.pt")
    
    logging.info(f'Training complete. Best AUROC: {best_val_auc:.4f} at epoch {best_epoch}')
    model.load_state_dict(best_model_params)
    return model
