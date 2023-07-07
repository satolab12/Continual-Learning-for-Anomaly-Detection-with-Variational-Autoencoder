import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from natsort import natsorted
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.labels = natsorted(os.listdir(root_dir))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.images = self._load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.images[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_images(self):
        images = []
        for label in self.labels:
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            label_idx = self.label_to_idx[label]
            image_files = os.listdir(label_dir)
            for image_file in image_files:
                image_path = os.path.join(label_dir, image_file)
                images.append((image_path, label_idx))

        return images

class Mnisttox(Dataset):
    def __init__(self, datasets ,labels:list):
        self.dataset = [datasets[i][0] for i in range(len(datasets))
                        if datasets[i][1] in labels ]
        self.labelset = [datasets[i][1] for i in range(len(datasets))
                        if datasets[i][1] in labels ]
        self.labels = labels
        self.len_oneclass = int(len(self.dataset)/10)

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, index):
        img = self.dataset[index]
        labelset = self.labelset[index]
        return img,labelset

class Parse():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int,default=64)
        parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--image_size', default=28)#32
        parser.add_argument('--n_channels', type=int, default=1)#3,1
        parser.add_argument('--z_dim', type=int, default=50)
        parser.add_argument('--gru_dim', type=int, default=100)#512,128,32
        parser.add_argument('--learning_rate', type=float, default=1e-3)#1e-2
        parser.add_argument('--cuda', type=bool, default=True)
        parser.add_argument('--flatten', type=bool, default=True)
        parser.add_argument('--adv_loss', default='hinge')#'hinge'wgan-gp

        self.args = parser.parse_args()


def roc(labels, scores, save_dir=None,plot_roc=False,plot_anomaly=False,normalize=False):
    """Compute ROC curve and ROC area for each class"""
    labels = labels#.cpu()
    scores = scores#.cpu()
    if normalize :
        max_score = np.amax(scores)
        scores = scores/max_score
    #print(scores)
    scores = np.nan_to_num(scores)
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    return roc_auc

