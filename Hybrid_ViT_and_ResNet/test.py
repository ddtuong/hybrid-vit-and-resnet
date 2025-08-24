from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import *
from engine import *
from models import *
from configurations import *
def main():
    #seed_everything
    seed_everything(43)
    
    # load dataset
    test_df = data_preprocessing(TEST_DIR)
    test_dataset = MyDataset(test_df, transforms_val)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )

    # load best model
    best_model = HybridViTResNet(vit_name=HYBRID["vit"], resnet_name=HYBRID["resnet"], num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    criterion = nn.CrossEntropyLoss() if MULTICLASSES else nn.BCEWithLogitsLoss() 

    print('============================== TESTING ==============================')
    test_loss, test_acc, test_pre, test_rec, _ = validate_one_epoch(best_model, test_loader, criterion, average=AVERAGE, device=DEVICE)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_pre}, Test Recall: {test_rec}")
    print('=====================================================================')

if __name__ == '__main__':
    main()