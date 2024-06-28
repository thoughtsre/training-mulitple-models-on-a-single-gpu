import os
from pathlib import Path
import datetime as dt
import logging

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
import torch.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

import lightning as L
from lightning.pytorch.loggers import CSVLogger


class MyClassifier(L.LightningModule):
    
    def __init__(self, n_classes=2, lr: float = 1e-3):
        
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder_weights = ResNet50_Weights
        self.encoder = resnet50(weights = self.encoder_weights.IMAGENET1K_V2)
        self.n_enc_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.encoder.eval()
        self.classifier = nn.Linear(self.n_enc_features, n_classes)
        self.loss = nn.CrossEntropyLoss()
        
        self.epoch_start_time = None
        
        return
    
    def forward(self, batch):
        
        e = self.encoder(batch)
        logits = self.classifier(e)
        
        return logits
    
    def on_train_epoch_start(self):
        
        self.epoch_start_time = dt.datetime.now()
        
        return
        
    def training_step(self, batch, batch_idx):
        
        image, y = batch
        
        e = self.encoder(image)
        logits = self.classifier(e)
        
        loss = self.loss(logits, y)
        
        return loss
    
    def on_train_epoch_end(self):
        
        self.log_dict({
            "train_epoch_duration": (dt.datetime.now() - self.epoch_start_time).total_seconds()
        })
        
        self.epoch_start_time = None
        
        return
    
    def validation_step(self, batch, batch_idx):
        
        image, y = batch
        
        e = self.encoder(image)
        logits = self.classifier(e)
        
        loss = self.loss(logits, y)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        
        return optimizer
    
    
class MyDataset(Dataset):
    
    def __init__(self, data_dir: Path):
        
        self.classes = os.listdir(data_dir)
        self.class_id = dict([(j, i) for i, j in enumerate(self.classes)])
        self.transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
        self.file_class = []
        
        # Normally I would just store the filename and read the data on-the-fly
        # But for this experiment I explicitly want all the data to be in-memory
        # right from the beginning.
        #
        # In this way, I don't run into bottlenecks with file IOs. The data is 
        # also small enough.
        for c in self.classes:
            
            for f in (data_dir / c).glob("*.jpg"):
                
                self.file_class.append((self.transforms(read_image(data_dir / f)), self.class_id[c]))
        
        return
    
    def __len__(self):
        
        return len(self.file_class)
    
    def __getitem__(self, idx):
        
        image, label = self.file_class[idx]
        
        return image, label
    

class MyDataModule(L.LightningDataModule):
    
    def __init__(self, root_data_dir: Path, batch_size: int = 4):
        
        super().__init__()
        self.save_hyperparameters()
        self.train_data_dir = root_data_dir / "train"
        self.val_data_dir = root_data_dir / "val"
        self.batch_size = batch_size
        
        return
    
    def setup(self, stage: str):
            
        self.train_dataset = MyDataset(self.train_data_dir)
        self.val_dataset = MyDataset(self.val_data_dir)
            
        return
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size = self.batch_size, 
                          shuffle = True, 
                          num_workers = 2)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size = self.batch_size, 
                          shuffle = True, 
                          num_workers = 2)
        
def train():
    
    max_epochs = int(os.environ.get("MAX_EPOCHS", 100))
    model_root_dir = os.environ.get("MODEL_ROOT_DIR", ".")
    data_dir = os.environ.get("ML_DATA_DIR", "/media/binghao/data/ml_data/hymenoptera_data")
    
    logger = CSVLogger(save_dir = Path(model_root_dir).parent, name=Path(model_root_dir).name)
    classifier = MyClassifier()
    datamodule = MyDataModule(Path(data_dir))
    trainer = L.Trainer(max_epochs = max_epochs, default_root_dir = model_root_dir, logger = logger)
    
    trainer.fit(classifier, datamodule=datamodule)
    
    return

if __name__ == "__main__":
    
    train()