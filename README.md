# hybrid-vit-and-resnet

This repository contains an implementation of a hybrid Vision Transformer (ViT) and ResNet architecture for image classification using PyTorch.
The model combines feature representations from both ViT and ResNet, and the final classification layer is replaced with a task-specific output layer for transfer learning. 

---

## Train and Test with transfer learning

### 🛠 Step 1: Configure Training Parameters
Before starting <small>**transfer learning**</small> training, first configure the training parameters in the <small>**`configurations.py`**<small> file:

This file contains all global settings related to dataset paths, model, training hyperparameters, and evaluation setup.  

---

### ⚙️ Example Configuration

```python
# DATASET 
TRAIN_DIR = "path/to/train"
VAL_DIR   = "path/to/val"
TEST_DIR  = "path/to/test"

# Save best model and training history
BEST_MODEL_PATH    = "model_name.pth"
MODEL_HISTORY_PATH = "history_name.pkl"

# Model setup
HYBRID = {
    "vit": "vit_b_16", # vit model you want
    "resnet": "resnet18" # resnet model you want
}

# Pretrained model
VIT_NAME2MODEL = {
    "vit_b_16": models.vit_b_16,
    "vit_b_32": models.vit_b_32,
    "vit_l_16": models.vit_l_16,
    "vit_l_32": models.vit_l_32,
}

RESNET_NAME2MODEL = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152
}

PRETRAINED = True
PROGRESS = True

# optimizer
LR = 1e-04
EPSILON = 1e-08
WEIGHT_DECAY = 1e-04
BETAS = (0.9, 0.999)

# model
NUM_CLASSES = 1
IMG_SIZE = 224
BATCH_SIZE = 5
N_EPOCHS = 2
MULTICLASSES = False # True = multi-class classification, False = binary

# Dataset labels
LABELS = {
    "APPLE": 0,
    "BANANA": 1,
    "ORANGE": 2
}
Id2LABELS = {v: k for k, v in LABELS.items()}

# Hardware configuration
NUM_WORKS = os.cpu_count()
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation configuration
AVERAGE = "micro" if MULTICLASSES else "binary"
```

### 📌 Notes
<small>**TRAIN_DIR, VAL_DIR, TEST_DIR**</small>
→ paths to your dataset folders. Each folder must contain subfolders (one per class).
Example:

```python
train/
├── APPLE/
├── BANANA/
└── ORANGE/

val/
├── APPLE/
├── BANANA/
└── ORANGE/

test/
├── APPLE/
├── BANANA/
└── ORANGE/
````
<small>**MODEL SELECTION**</small>
```python
HYBRID = {
    "vit": "vit_b_16",     # specify the ViT variant for the hybrid model
    "resnet": "resnet18"   # specify the ResNet variant for the hybrid model
}
```

<small>**NUM_CLASSES**</small>
→ must equal the number of subfolders (classes) in the dataset.

<small>**LABELS**</small>
→ dictionary mapping folder name to class index.
Example:
```python
LABELS = {"APPLE": 0, "BANANA": 1, "ORANGE": 2}
```

<small>**DEVICE**</small>
→ automatically detects cuda if available, otherwise falls back to cpu.

<small>**AVERAGE**</small>
→ controls evaluation metrics:
- "micro" for multi-class
- "binary" for binary classification

### 🏋️ Step 2: Train with Transfer Learning

Once the configuration file (`configurations.py`) is ready, you can start training by editing the <small>**`train.py`**<small> file.

---

### 📂 Example: `train.py`

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle as pkl
from datasets import *
from engine import *
from models import *
from configurations import *

def main():
    # 1. Set random seed for reproducibility
    seed_everything(43)
    
    # 2. Load datasets
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

    # 3. Load model & optimizer
    model = HybridViTResNet(vit_name=HYBRID["vit"], resnet_name=HYBRID["resnet"], num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss() if MULTICLASSES else nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adamax(model.parameters(), 
                             lr=LR,   # learning rate 
                             betas=BETAS,  # moment coefficients
                             eps=EPSILON,  # to avoid division by zero
                             weight_decay=WEIGHT_DECAY)   # L2 regularization 

    # 4. Training loop
    print('============================== TRAINING ==============================')
    history = fit(
        model, 
        train_loader, 
        valid_loader, 
        N_EPOCHS, 
        criterion, 
        optimizer, 
        average=AVERAGE, 
        device=DEVICE
    )
    print('======================================================================')

    # 5. Save training history
    with open(MODEL_HISTORY_PATH, 'wb') as file:
        pkl.dump(history, file)

if __name__ == '__main__':
    main()
```

### 🧪 Step 3: Testing (Evaluation)

After training, you can evaluate the model on the **test dataset** to measure its generalization performance.  
The evaluation includes **Loss, Accuracy, Precision, and Recall**.

---

### 📂 Example: `test.py`

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import *
from engine import *
from models import *
from configurations import *

def main():
    # 1. Set random seed
    seed_everything(43)
    
    # 2. Load test dataset
    test_df = data_preprocessing(TEST_DIR)
    test_dataset = MyDataset(test_df, transforms_val)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )

    # 3. Load best trained model
    best_model = HybridViTResNet(vit_name=HYBRID["vit"], resnet_name=HYBRID["resnet"], num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))

    criterion = nn.CrossEntropyLoss() if MULTICLASSES else nn.BCEWithLogitsLoss() 

    # 4. Run evaluation
    print('============================== TESTING ==============================')
    test_loss, test_acc, test_pre, test_rec, _ = validate_one_epoch(
        best_model, 
        test_loader, 
        criterion, 
        average=AVERAGE, 
        device=DEVICE
    )
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_pre}, Test Recall: {test_rec}")
    print('=====================================================================')

if __name__ == '__main__':
    main()
```

## 📌 Change Model for Testing
As in training, you can switch the model easily:

```python
# Example: Hybrid vit_b_16 and resnet101
best_model = HybridViTResNet(vit_name="vit_b_16", resnet_name="resnet101", num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

# Example: Hybrid vit_l_16 and resnet101
best_model = HybridViTResNet(vit_name="vit_l_16", resnet_name="resnet101", num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)

```

### Installation
```python
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

