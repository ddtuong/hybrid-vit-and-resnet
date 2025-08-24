from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import random
import torch
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from configurations import BEST_MODEL_PATH, MULTICLASSES

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, train_loader, criterion, optimizer, average='binary', device='cpu'):
    model.train()
    losses = 0.0
    accuracy_scores = 0.0
    precision_scores = 0.0
    recall_scores = 0.0

    for (data, target) in tqdm(train_loader, total=len(train_loader)):
        target = target.long() if MULTICLASSES else target.unsqueeze(1).float()
        if device == "cuda":
            data, target = data.cuda(), target.cuda()
        output = model(data)
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses += loss.item()
        if MULTICLASSES:
            output = torch.argmax(output, dim=1)
            output = output.detach().cpu().numpy().astype(int).flatten()
        else:
            output = torch.sigmoid(output)
            output = (output > 0.5).detach().cpu().numpy().astype(int).flatten()

        target = target.detach().cpu().numpy().astype(int).flatten()

        accuracy_scores += metrics.accuracy_score(target, output)
        precision_scores += metrics.precision_score(target, output, average=average, zero_division=0)
        recall_scores += metrics.recall_score(target, output, average=average, zero_division=0)

    return losses/len(train_loader), accuracy_scores/len(train_loader), precision_scores/len(train_loader), recall_scores/len(train_loader) 

def validate_one_epoch(model, valid_loader, criterion, average='binary', device='cpu'):
    model.eval()
    with torch.no_grad():
        losses = 0.0
        preds = []
        targets = []
        for (data, target) in tqdm(valid_loader, total=len(valid_loader)):
            target = target.long() if MULTICLASSES else target.unsqueeze(1).float()
            if device == "cuda":
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            losses += loss.item()

            if MULTICLASSES:
                output = torch.argmax(output, dim=1)
                output = output.detach().cpu().numpy().astype(int).flatten()
            else:
                output = torch.sigmoid(output)
                output = (output > 0.5).detach().cpu().numpy().astype(int).flatten()

            target = target.detach().cpu().numpy().astype(int).flatten()
          
            preds.append(output)
            targets.append(target)
        
        # 1D
        preds = np.hstack(preds)   
        targets = np.hstack(targets)

        accuracy_score = metrics.accuracy_score(targets, preds)
        precision_score = metrics.precision_score(targets, preds, average=average, zero_division=0)
        recall_score = metrics.recall_score(targets, preds, average=average, zero_division=0)
        confusion_matrix = metrics.confusion_matrix(targets, preds)
    
        return losses/len(valid_loader), accuracy_score, precision_score, recall_score, confusion_matrix


def fit(model, train_loader, valid_loader, epochs, criterion, optimizer, average='binary', device='cpu'):
    train_losses = []
    train_accs = []
    train_pres = []
    train_recs = []
    
    valid_losses = []
    valid_accs = []
    valid_pres = []
    valid_recs = []
    best_loss = 1000
    
    for epoch in range(1, epochs+1):
        gc.collect()
        loss, acc, pre, rec = train_one_epoch(model, train_loader, criterion, optimizer, average=average, device=device)
        valid_loss, valid_acc, valid_pre, valid_rec, _ = validate_one_epoch(model, valid_loader, criterion, average=average, device=device)
        print(f"Training loss: {loss}, Accuracy: {acc}, Precision: {pre}, Recall: {rec}")
        print(f"Validation loss: {valid_loss}, Validation Accuracy: {valid_acc}, Validation Precision: {valid_pre}, Validation Recall: {valid_rec}")
        
        # save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            
        train_losses.append(loss)
        train_accs.append(acc)
        train_pres.append(pre)
        train_recs.append(rec)

        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_pres.append(valid_pre)
        valid_recs.append(valid_rec)

    return {
        "train_loss": train_losses,
        "train_accuracy": train_accs,
        "train_precision": train_pres,
        "train_recall": train_recs,
        "valid_loss": valid_losses,
        "valid_accuracy": valid_accs,
        "valid_precision": valid_pres,
        "valid_recall": valid_recs
    }
        