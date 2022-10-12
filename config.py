import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/home/ml/Hiren/Data/Train"
VAL_DIR = "/home/ml/Hiren/Data/Val"
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
CYCLE = 2
IDENTITY=1
NUM_WORKERS = 4
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True

transforms = A.Compose(
    [
        #A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.7),
        A.VerticalFlip(p=0.6),
        A.RandomRotate90(p=0.8),
        #A.RandomCrop(p=0.4),
        #A.Normalize(mean=[0.6448, 0.3841, 0.2019], std=[0.1238, 0.1057, 0.0832], max_pixel_value=255),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image1": "image"}
)