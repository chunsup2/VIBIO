import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from network_lumpy.CNN_IO import BinaryClassifier
from network_lumpy.VAE import VIBCNN, VIBHO
from network_lumpy.HO import SLNNHO

from dataloader import LumpyDataset
from torch.utils.data import DataLoader, Subset 
import numpy as np
import argparse
import wandb
from wandb import Image
from utils import load_model

from sklearn.metrics import roc_auc_score, RocCurveDisplay  # Modified import
import matplotlib.pyplot as plt
import sys

import torch

seed = 42

import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)   # 固定权重初始化，减少大差异


def train(args):

    torch.backends.cudnn.benchmark = True

    # Initialize wandb
    data_name = args.data.replace('lumpy_background', '')
    
    # Create descriptive run name including cls_type and KL weight
    if args.cls_type in ['VIBHO', 'VIBCNN']:
        run_name = f'{args.cls_type}_kl{args.kl}_prop{args.proporation}{data_name}'
    else:
        run_name = f'{args.cls_type}_prop{args.proporation}{data_name}'
    
    wandb.init(project='VIB-Lumpy', name=run_name, mode='online', 
               settings=wandb.Settings(_service_wait=120))
    
    device = torch.device(args.device)

    if not os.path.exists(os.path.join(args.save_model_path)):
        os.makedirs(os.path.join(args.save_model_path))
    
    if args.cls_type == 'ResNet':
        cls_model = ResNetX(args.depth, num_classes=2).to(device)
    elif args.cls_type == 'CNN':
        cls_model = BinaryClassifier(args.depth, num_classes=2).to(device)
    elif args.cls_type == 'HO':
        cls_model = SLNNHO().to(device)
    elif args.cls_type == 'VIBHO':
        cls_model = VIBHO().to(device)
    elif args.cls_type == 'VIBCNN':
        cls_model = VIBCNN(args.depth, args.z_dim, num_classes=2).to(device)

    cls_model = nn.DataParallel(cls_model)

    # Optimizer and scheduler for cls_model
    if args.depth != 1:
        if args.cycoptim == True:
            optimizer = optim.Adam(list(cls_model.parameters()), lr=args.lr)
            scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr, max_lr=1e-4, step_size_up=10, cycle_momentum=False)
            scheduler_flag = 1
        else:
            optimizer = optim.Adam(list(cls_model.parameters()), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            scheduler_flag = 0

    else:
        optimizer = optim.Adam(list(cls_model.parameters()), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        scheduler_flag = 0

    # Loss functions
    criterion = nn.MSELoss().to(device)
    cls_criterion = nn.CrossEntropyLoss().to(device)  # CrossEntropyLoss expects integer labels
    
    cls_criterion2 = nn.BCEWithLogitsLoss().to(device)  # Binary Cross Entropy Loss for binary classification

    train_dataset = LumpyDataset(args.train_image_path, args.proporation)
    test_dataset = LumpyDataset(args.test_image_path, proportion=1.0)    
    val_dataset = LumpyDataset(args.val_image_path, proportion=1.0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=4)

    start_epoch = 0
    best_loss = torch.tensor(float('inf'))

    for epoch in tqdm(range(start_epoch, args.epochs),
                  total=args.epochs - start_epoch, desc="epoch"):  
  
        cls_model.train()
        for i, (image, task_label) in enumerate(tqdm(train_dataloader)):
            image, task_label = image.to(device),task_label.to(device)
            
            # Remove one-hot encoding; ensure task_label is LongTensor
            task_label = task_label.long()

            feats = image
          
            loss = torch.tensor(0.0).to(device)
            ho_loss = torch.tensor(0.0).to(device)
            kl_loss = torch.tensor(0.0).to(device)
            recon_loss = torch.tensor(0.0).to(device)
            cls_loss = torch.tensor(0.0).to(device)
            if args.cls_type == 'CNN' or args.cls_type == 'ResNet':

                cls = cls_model(feats)
                loss = cls_criterion(cls, task_label)   

            elif args.cls_type == 'HO':

                task_label_tmp = task_label.view(-1,1,1,1)

                g0 = 2 * torch.mean((1 - task_label_tmp) * feats, dim = 0, keepdim=True)
                g1 = 2 * torch.mean(task_label_tmp * feats, dim = 0, keepdim=True)
                delta_g = g1 - g0

                cls1 = cls_model(feats-(g0).repeat(feats.shape[0],1,1,1))
                cls2 = cls_model(feats-(g1).repeat(feats.shape[0],1,1,1))
                delta_cls = cls_model(delta_g)

                ho_loss = torch.mean((1 - task_label) * cls1 ** 2 + task_label * cls2 ** 2) \
                                    - 2 * delta_cls 
                loss = ho_loss
                # cls = cls_model(feats).squeeze()
                # print(cls.shape, task_label.shape)
                # loss = cls_criterion2(cls, task_label.float())  # Use BCE loss for binary classification
            
            elif args.cls_type == 'VIBHO':

                t, mu, logvar = cls_model(feats)
                t = t.view(-1, 1, 1, 1)

                task_label_tmp = task_label.view(-1,1,1,1)

                g0 = 2 * torch.mean((1 - task_label_tmp) * feats, dim = 0, keepdim=True)
                g1 = 2 * torch.mean(task_label_tmp * feats, dim = 0, keepdim=True)
                delta_g = g1 - g0

                t1, mu1, logvar1 = cls_model(feats-(g0).repeat(feats.shape[0],1,1,1))
                t2, mu2, logvar2 = cls_model(feats-(g1).repeat(feats.shape[0],1,1,1))
                t_delta, mu_delta, logvar_delta = cls_model(delta_g)

                ho_loss = torch.mean((1 - task_label) * t1 ** 2 + task_label * t2 ** 2) \
                                    - 2 * t_delta   

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

                loss = ho_loss + args.kl * kl_loss
            
            elif args.cls_type == 'VIBCNN':
               
                # one hot the task_label: 0 -> [0, 1], 1 -> [1, 0]
     
                label_one_hot = F.one_hot(task_label, num_classes=2).float()
       
                t, mu, logvar, recon = cls_model(feats, label=label_one_hot)
                # recon_loss = F.mse_loss(recon, feats)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                cls_loss = cls_criterion(t, task_label)

                loss = cls_loss + args.kl * kl_loss

                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler_flag == 1:
                scheduler.step()

            wandb.log({"Train Loss": loss.item(),
                        "Train Recon Loss": recon_loss.item(),
                        "Train KL Loss": kl_loss.item(),
                        "Train HO Loss": ho_loss.item(),
                        "Train Class Loss": cls_loss.item(),
                        })

        if scheduler_flag == 0:
            scheduler.step()

        # Test
        cls_model.eval()
        # total_loss = 0
        # correct = 0
        # total = 0
        best_auc = 0.5
        best_epoch = 0
        true_labels = []
        predicted_probs = []
        predicted_labels = []  # To store predicted class indices
        with torch.no_grad():
            for i, (image, task_label) in enumerate(tqdm(val_dataloader)):
                image, task_label = image.to(device), task_label.to(device)

                task_label = task_label.long()

                feats = image


                if args.cls_type == 'CNN' or args.cls_type == 'ResNet':
                    cls = cls_model(feats)
                elif args.cls_type == 'HO':
                    cls = cls_model(feats)
                elif args.cls_type == 'VIBHO':
                    t, mu, logvar = cls_model(feats)
                    cls = mu
                elif args.cls_type == 'VIBCNN':
                    t, mu, logvar, recon = cls_model(feats, mode='test')
                    cls = t


                if args.cls_type == 'CNN' or args.cls_type == 'ResNet' or args.cls_type == 'VIBCNN':
                    cls_probs = F.softmax(cls, dim=1)
                    _, predicted = torch.max(cls_probs, 1)
                    predicted_labels.extend(predicted.detach().cpu().numpy())
                    probs_for_true_labels = cls_probs[torch.arange(cls_probs.size(0)), 0]
                    predicted_probs.extend(probs_for_true_labels.detach().cpu().numpy())
                    # total += task_label.size(0)
                    # correct += (predicted == task_label).sum().item()

                if args.cls_type == 'HO' or args.cls_type == 'VIBHO':
                    test_stat = cls.squeeze()
                    predicted = (cls > 0.5).long()
                    predicted_labels.extend(predicted.detach().cpu().numpy())
                    predicted_probs.extend(test_stat.detach().cpu().numpy())
               
                true_labels.extend(task_label.detach().cpu().numpy())
                                
                if i == 0:
                    image_ = image[0].squeeze(0).detach().cpu().numpy()
                    wandb.log({
                        # "Test Loss": loss.item(),
                        "Images": [wandb.Image(image_, caption="Object"),
                                   ]
                    })
                
        # Calculate the accuracy
        # if args.cls_type == 'CNN' or args.cls_type == 'ResNet' or args.cls_type == 'VIBCNN':
        #     accuracy = 100 * correct / total
        #     wandb.log({"Test Accuracy": accuracy})

        # Convert true_labels and predicted_probs to appropriate format
   
        num_classes = 2

        true_labels_array = np.array(true_labels)
        predicted_probs_array = np.array(predicted_probs).reshape(-1, 1)

        # One-hot encode true labels for ROC AUC calculation
        true_labels_one_hot = np.eye(num_classes)[true_labels_array]

        # Calculate macro-average ROC AUC
        roc_auc = roc_auc_score(true_labels_one_hot, predicted_probs_array, average='macro', multi_class='ovo')
        if roc_auc < 0.5:
            roc_auc = 1 - roc_auc
        wandb.log({"Test AUC": roc_auc})

        # # Plot ROC Curve for each class
        # plt.figure()
        # for class_idx in range(num_classes):
        #     RocCurveDisplay.from_predictions(
        #         true_labels_one_hot[:, class_idx],
        #         predicted_probs_array[:, class_idx],
        #         name=f"Class {class_idx}"
        #     )
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve')
        # plt.legend(loc="lower right")
        # wandb.log({"ROC Curve": wandb.Image(plt)})
        # plt.close()

        if epoch <= 20:
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_epoch = epoch
                # Save the best model
                if args.cls_type == 'VIBHO' or args.cls_type == 'VIBCNN':
                    torch.save(cls_model.module.state_dict(), os.path.join(args.save_model_path) + '/clsnet_{}_kl{}_lr{}_kl{}_depth{}_z{}_e{}.pth'.format(args.cls_type, args.kl, args.lr, args.kl, args.depth, args.z_dim, args.epochs))
                else:
                    torch.save(cls_model.module.state_dict(), os.path.join(args.save_model_path) + '/clsnet_{}_lr{}_depth{}_z{}_e{}.pth'.format(args.cls_type, args.lr, args.depth, args.z_dim, args.epochs))
        if epoch > 20:
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_epoch = epoch
                # Save the best model
                if args.cls_type == 'VIBHO' or args.cls_type == 'VIBCNN':
                    torch.save(cls_model.module.state_dict(), os.path.join(args.save_model_path) + '/clsnet_{}_kl{}_lr{}_kl{}_depth{}_z{}_e{}.pth'.format(args.cls_type, args.kl, args.lr, args.kl, args.depth, args.z_dim, args.epochs))
                else:
                    torch.save(cls_model.module.state_dict(), os.path.join(args.save_model_path) + '/clsnet_{}_lr{}_depth{}_z{}_e{}.pth'.format(args.cls_type, args.lr, args.depth, args.z_dim, args.epochs))
            else:
                if (epoch - best_epoch) > 12:
                    print("Early stopping at epoch ", epoch)
                    print("Best epoch is ", best_epoch)
                    break
    

if __name__ == '__main__':

    # gpu 
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id 
    
    # Remove the first argument (GPU ID) from sys.argv so argparse doesn't see it
    if len(sys.argv) > 1:
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--depth', type=int, default=6, help='1-10')
    parser.add_argument('--lpips', type=bool, default=False)
    parser.add_argument('--cycoptim', type=bool, default=False)
    parser.add_argument('--cls_type', type=str, default='VIBHO', help='CNN, ResNet, HO, VIBHO')

    parser.add_argument('--kl', type=float, default=0.001, help='0.1 for VIBHO, 0.0005 for VIBCNN')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for training')
    parser.add_argument('--proporation', type=float, default=1.0, help='0.001, 0.01, 0.1, 0.2, 0.5, 1.0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--data', type=str, default='lumpy_background_A0.2_std30_num200000')
    parser.add_argument('--train_image_path', type=str, default=None)
    parser.add_argument('--test_image_path', type=str, default=None)
    parser.add_argument('--val_image_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default=None)

    # Paths and settings
    args = parser.parse_args()
    
    # Include depth for CNN and VIBCNN models, but not for HO and VIBHO (single-layer networks)
    if args.cls_type in ['CNN', 'VIBCNN']:
        args.save_model_path = 'checkpoint/{}/{}/depth_{}'.format(args.data, args.proporation, args.depth)
    else:  # HO and VIBHO don't use depth
        args.save_model_path = 'checkpoint/{}/{}'.format(args.data, args.proporation)   
    args.train_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/train/data.h5'.format(args.data)
    args.test_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/test/data.h5'.format(args.data)
    args.val_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/val/data.h5'.format(args.data)
    
    train(args)
