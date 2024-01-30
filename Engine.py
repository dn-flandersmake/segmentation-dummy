import os
import shutil
import torch

from datasets import get_dataset
from models import get_model
from loss_fns import get_loss

from tqdm import tqdm
import pathlib
from utils import plot


class Engine:
    def __init__(self, config, device=None):

        self.config = config
        
        # set device
        self.device = (
            device
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # create output dir & copy config file
        os.makedirs(config['save_dir'], exist_ok=True)
        os.system(f"cp {pathlib.Path(__file__).parent.absolute()}/configs/{config['config']}.py {config['save_dir']}")

        # train dataset
        train_dataset = get_dataset(
            config['train_dataset']['name'], config['train_dataset']['kwargs']
        )
        self.train_dataset_it = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['train_dataset']['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=config['train_dataset']['workers'],
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # model
        self.model = get_model(
            config['model']['name'], config['model']['kwargs']
        ).to(self.device)

        # loss
        self.loss_fn = get_loss(
            config['loss']['name'], config['loss']['kwargs']
        )

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])


    def forward(self, sample):
        images = sample["image"].to(self.device)
        labels = sample["label"].to(self.device)

        # needs to be this format for dice/bce loss
        labels = (labels > 0).unsqueeze(1).float()

        pred = self.model(images)
        loss = self.loss_fn(pred, labels)

        return pred, loss

    def train_step(self, epoch):

        # define meters
        loss_meter = 0

        self.model.train()
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            pred, loss = self.forward(sample)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_meter += loss.item()

        # display last samples
        plot.plot(
            sample['image'][0],
            sample['label'][0],
            pred[0] > 0,
            path=os.path.join(self.config['save_dir'], f'epoch_{epoch}.jpg')
        )

        return loss_meter / (i + 1)

    def train(self):
        
        for epoch in range(self.config['n_epochs']):

            print(f'Starting epoch {epoch}')

            train_loss = self.train_step(epoch)

            print(f"==> train loss: {train_loss}")
