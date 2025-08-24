from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl
from datasets import *
from engine import *
from models import *
from configurations import *

def main():
    #seed_everything
    seed_everything(43)
    
    # load dataset
    train_df = data_preprocessing(TRAIN_DIR)
    valid_df = data_preprocessing(VAL_DIR)
    
    train_dataset = MyDataset(train_df, transforms_train)
    valid_dataset = MyDataset(valid_df, transforms_val)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )

    # load model and loss
    model = HybridViTResNet(vit_name=HYBRID["vit"], resnet_name=HYBRID["resnet"], num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss() if MULTICLASSES else nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adamax(model.parameters(), 
                         lr=LR,   # learning rate 
                         betas=BETAS,  # moment coefficients
                         eps=EPSILON,  # to avoid division by zero
                         weight_decay=WEIGHT_DECAY)   # L2 regularization 

    # fit
    print('============================== TRAINING ==============================')
    history = fit(model, train_loader, valid_loader, N_EPOCHS, criterion, optimizer, average=AVERAGE, device=DEVICE)
    print('======================================================================')
    # saving history
    with open(MODEL_HISTORY_PATH, 'wb') as file:
        pkl.dump(history, file)

if __name__ == '__main__':
    main()