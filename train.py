import os
import torch.nn as nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import random
import numpy as np
from tqdm import tqdm
import math
from model import MRFNet


class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.clear_dir = os.path.join(root_dir, 'clear')
        self.cloud_dir = os.path.join(root_dir, 'cloud')
        self.image_filenames = [img for img in os.listdir(self.cloud_dir) if
                                img.endswith('.png') or img.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize([0.34807074], [0.11415784])])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        cloud_img_name = os.path.join(self.cloud_dir, self.image_filenames[idx])
        clear_img_name = os.path.join(self.clear_dir, self.image_filenames[idx])
        cloud_img = Image.open(cloud_img_name).convert('L')
        clear_img = Image.open(clear_img_name).convert('L')
        clear_img = self.transform(clear_img)
        cloud_img = self.transform(cloud_img)
        sample = {'cloud_img': cloud_img, 'clear_img': clear_img}
        return sample


def train(model, train_dataloader, optimizer, criterion, device, save_dir, save_cycle, epochs, resume=None,
          crop_size=240):
    if resume is not None:
        if not os.path.exists(resume):
            raise FileNotFoundError(f'Checkpoint {resume} not found.')
        else:
            print(f'Resume from {resume}')
            model.load_state_dict(torch.load(args.resume))
    else:
        print('Train from scratch...')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_loss = float("inf")
    for epoch in range(epochs):
        losses = {'loss': [], 'loss1': [], 'loss2': []}
        for batch in tqdm(train_dataloader):
            cloud_imgs = batch['cloud_img'].to(device)
            clear_imgs = batch['clear_img'].to(device)
            _, _, height, width = cloud_imgs.shape
            if crop_size >= width or crop_size >= height:
                raise ValueError('crop_size must be less than img_size')
            max_coordinate = min(width, height) - crop_size
            x = random.randint(0, max_coordinate)
            y = random.randint(0, max_coordinate)
            cropped_cloud_imgs = cloud_imgs[:, :, y:y + crop_size, x:x + crop_size]
            res = model(cloud_imgs)
            cropped_res = model(cropped_cloud_imgs)
            loss1 = criterion(res, clear_imgs)
            loss2 = criterion(res[:, :, y:y + crop_size, x:x + crop_size], cropped_res)
            loss = 0.8 * loss1 + 5 * loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if math.isnan(loss.item()):
                raise ValueError('loss is nan while training')
            losses['loss'].append(loss.item())
            losses['loss1'].append(loss1.item())
            losses['loss2'].append(loss2.item())
        print('Epoch:{}/{} | Loss:{:.4f} Loss1:{:.4f} Loss2:{:.4f}'
              .format(epoch + 1, epochs, np.mean(losses['loss']), np.mean(losses['loss1']), np.mean(losses['loss2'])))
        if np.mean(losses['loss']) <= best_loss:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            best_loss = np.mean(losses['loss'])
            print('Saving best model...')
        if (epoch + 1) % save_cycle == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, '{}.pth'.format(epoch + 1)))
    print('\nTrain Complete.\n')
    torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))


def main(args):
    dataset = CloudRemovalDataset(root_dir=args.data_dir)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')
    model = MRFNet().to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 betas=(0.9, 0.999),
                                 eps=1e-08)
    optimizer.zero_grad()
    train(model, train_dataloader, optimizer, criterion, device, args.save_dir, args.save_cycle, args.epochs,
          args.resume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--data_dir', type=str, default=r'./dataset/res', help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default=r'./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--save_cycle', type=int, default=5, help='Cycle of saving checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size of each data batch')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    args = parser.parse_args()
    main(args)
