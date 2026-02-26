import os
os.environ["WANDB__SERVICE_WAIT"] = "120"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from network_lumpy.UNet import UNet
from network_lumpy.CNN_IO import BinaryClassifier
from network_lumpy.VAE import VIBCNN, VIBHO
from network_lumpy.HO import SLNNHO
from network_lumpy.ResNet_IO import ResNetX
from dataloader import LumpyDataset
from dataloader import MRIDataset
from torch.utils.data import DataLoader, Subset 
import numpy as np
import argparse
import wandb
from wandb import Image
from utils import load_model, normal_IO_train_torch, normal_IO_test_torch, normal_IO_train_torch1, normal_IO_train_torch2  # Modified import
from test import normal_IO_train, normal_IO_test  # Modified import
import sys
from sklearn.metrics import roc_auc_score, RocCurveDisplay  # Modified import
import matplotlib.pyplot as plt


seed = 42

import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)   # 固定权重初始化，减少大差异


def train(args):

    torch.backends.cudnn.benchmark = True

    # Initialize wandb
    data_name = args.data.replace('_num_signals', 'ema')
    wandb.init(project='VIB-MRI', name = '{}'.format(args.proporation) + args.train_type + args.cls_type + data_name, mode='online', 
               settings=wandb.Settings(_service_wait=120))
    
    device = torch.device(args.device)

    if args.train_type == 'recon':
        model = UNet(n_channels=1, n_classes=1, bilinear=True).to(device)
        
        # Load the model and use multi-GPU
        model = nn.DataParallel(model)
        print(device)

        load_model(model, os.path.join(args.save_model_path, args.srtype) + '/srnet.pth')

    if not os.path.exists(os.path.join(args.save_model_path, args.srtype)):
        os.makedirs(os.path.join(args.save_model_path, args.srtype))
    
    # sks_3_c3_num_signals -> num_classes = 3  "detect 'c3"
    # if 'c3' in args.data:
    #     if args.cls_type == 'ResNet':
    #         cls_model = ResNetX(args.depth, num_classes=3).to(device)
    #     elif args.cls_type == 'CNN':
    #         cls_model = BinaryClassifier(args.depth, num_classes=3).to(device)
    #     elif args.cls_type == 'VIBCNN':
    #         cls_model = VIBCNN(args.depth, num_classes=3).to(device)
    # if 'c2' in args.data:
    #     if args.cls_type == 'ResNet':
    #         cls_model = ResNetX(args.depth, num_classes=2).to(device)
    #     elif args.cls_type == 'CNN':
    #         cls_model = BinaryClassifier(args.depth, num_classes=2).to(device)
    #     elif args.cls_type == 'HO':
    #         cls_model = SLNNHO().to(device)
    #     elif args.cls_type == 'VIBHO':
    #         cls_model = VIBHO().to(device)
    #     elif args.cls_type == 'VIBCNN':
    #         cls_model = VIBCNN(args.depth, num_classes=2).to(device)

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
        if 'c2' in args.data:
            optimizer = optim.Adam(list(cls_model.parameters()), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            scheduler_flag = 0
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
       
    # train_dataset = MRIDataset(args.train_image_path, args.proporation)
    # test_dataset = MRIDataset(args.test_image_path, proportion=1.0)    
    # val_dataset = MRIDataset(args.val_image_path, proportion=1.0)

    train_dataset = LumpyDataset(args.train_image_path, args.proporation)
    test_dataset = LumpyDataset(args.test_image_path, proportion=1.0)    
    val_dataset = LumpyDataset(args.val_image_path, proportion=1.0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=8)

    start_epoch = 0
    best_loss = torch.tensor(float('inf'))
    best_auc = 0.5
    best_epoch = 0

    for epoch in tqdm(range(start_epoch, args.epochs)):  
        if args.train_type == 'recon':
            model.eval()
        cls_model.train()
        mu_ema = np.zeros((1, args.z_dim))
        Kinv_ema = np.zeros((args.z_dim, args.z_dim))
        s_ema = np.zeros((1, args.z_dim))
        K_ema = np.zeros((args.z_dim, args.z_dim))
        for i, (image, task_label) in enumerate(tqdm(train_dataloader)):
            image, task_label = image.to(device), task_label.to(device)
            
            # Remove one-hot encoding; ensure task_label is LongTensor
            task_label = task_label.long()

            # if args.train_type == 'recon':
            #     feats = recon
            # elif args.train_type == 'measure':
            #     feats = measure
            # elif args.train_type == 'cls':
            feats = image
            # else:
            #     raise ValueError(f"Unknown train_type {args.train_type}")
            
            loss = torch.tensor(0.0).to(device)
            ho_loss = torch.tensor(0.0).to(device)
            kl_loss = torch.tensor(0.0).to(device)
            # recon_loss = torch.tensor(0.0).to(device)
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
                # if 'c2' in args.data:
                label_one_hot = F.one_hot(task_label, num_classes=2).float()
                # elif 'c3' in args.data:
                #     label_one_hot = F.one_hot(task_label, num_classes=3).float()

                t, mu, logvar, recon = cls_model(feats, label=label_one_hot)
                # recon_loss = F.mse_loss(recon, feats)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                # cls_loss = cls_criterion(t, task_label)
                mu0_mean, s, Kinv, K = normal_IO_train_torch(mu, task_label)
                # mu0_mean, s, Kinv = normal_IO_train_torch1(mu, logvar, task_label)
                # mu0_mean, s, Kinv = normal_IO_train_torch2(mu, logvar, task_label)
                lambda_full = normal_IO_test_torch(mu, mu0_mean, s, Kinv)
                # convert lambda_full to tensor and maximize the distance between the two classes
                lambda_0 = lambda_full[task_label == 0]
                lambda_1 = lambda_full[task_label == 1]
                cls_loss = -torch.mean(lambda_1) + torch.mean(lambda_0)
                cls_loss = -torch.log(-cls_loss + 1e-8)  # log to avoid negative values

                loss = args.ioloss * cls_loss + args.kl * kl_loss

                mu_ema = mu0_mean
                Kinv_ema = Kinv
                s_ema = s
                K_ema = K

                   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler_flag == 1:
                scheduler.step()

            wandb.log({"Train Loss": loss.item(),
                        # "Train Recon Loss": recon_loss.item(),
                        "Train KL Loss": args.kl * kl_loss.item(),
                        "Train HO Loss": ho_loss.item(),
                        "Train Class Loss": args.ioloss * cls_loss.item(),
                        })

        if scheduler_flag == 0:
            scheduler.step()

         # Test
        cls_model.eval()
        # total_loss = 0
        # correct = 0
        # total = 0
        
        true_labels = []
        predicted_probs = []
        predicted_labels = []  # To store predicted class indices
        mu_all = np.zeros((1, args.z_dim))
        label_all = np.zeros((1, 1))
        mu_all_train = np.zeros((1, args.z_dim))
        label_all_train = np.zeros((1, 1))


        with torch.no_grad():
            for i, (image, task_label) in enumerate(tqdm(val_dataloader)):
                image, task_label = image.to(device), task_label.to(device)

                task_label = task_label.long()

                # if args.train_type == 'recon':
                #     feats = recon
                # elif args.train_type == 'measure':
                #     feats = measure
                # elif args.train_type == 'cls':
                feats = image
                # else:
                #     raise ValueError(f"Unknown train_type {args.train_type}")


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

                    labels_ = task_label.view(-1, 1)
                    mu_all = np.concatenate((mu_all, mu.detach().cpu().numpy()), axis=0)
                    label_all = np.concatenate((label_all, labels_.detach().cpu().numpy()), axis=0)


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
                    predicted = (test_stat > 0.5).to(torch.int64)  # Using to(torch.int64) instead of long()
                    predicted_labels.extend(predicted.detach().cpu().numpy())
                    predicted_probs.extend(test_stat.detach().cpu().numpy())
                                
                
                true_labels.extend(task_label.detach().cpu().numpy())
                                
                if i == 0:
                    image_ = image[0].squeeze(0).detach().cpu().numpy()
                    # measure_ = measure[0].squeeze(0).detach().cpu().numpy()
                    wandb.log({
                        # "Test Loss": loss.item(),
                        "Images": [wandb.Image(image_, caption="Object")
                                   ]
                    })
                    if args.train_type == 'recon':
                        recon_ = recon[0].squeeze(0).detach().cpu().numpy()
                        wandb.log({
                            "Images": [wandb.Image(recon_, caption="Reconstruction")]
                        })
                
        # Calculate the accuracy
        # if args.cls_type == 'CNN' or args.cls_type == 'ResNet' or args.cls_type == 'VIBCNN':
        #     accuracy = 100 * correct / total
        #     wandb.log({"Test Accuracy": accuracy})

        # Convert true_labels and predicted_probs to appropriate format
        # if 'c3' in args.data:
        #     num_classes = 3
        # if 'c2' in args.data:
        num_classes = 2

        true_labels_array = np.array(true_labels)
        predicted_probs_array = np.array(predicted_probs).reshape(-1, 1)

        # One-hot encode true labels for ROC AUC calculation
        true_labels_one_hot = np.eye(num_classes)[true_labels_array]

        if args.cls_type != 'VIBCNN':
            # Calculate macro-average ROC AUC
            roc_auc = roc_auc_score(true_labels_array, predicted_probs_array)

            if roc_auc < 0.5:
                roc_auc = 1 - roc_auc
            wandb.log({"Test AUC": roc_auc})

        if args.cls_type == 'VIBCNN':
            mu_all = mu_all[1:]
            label_all = label_all[1:]
            # to numpy
            mu_ema_ = mu_ema.detach().cpu().numpy()
            s_ema_ = s_ema.detach().cpu().numpy()
            Kinv_ema_ = Kinv_ema.detach().cpu().numpy()
            K_ema_ = K_ema.detach().cpu().numpy()
            lambda_ = normal_IO_test(mu_all, mu_ema_, s_ema_, Kinv_ema_)
            scores = lambda_
            label_all = label_all.flatten()
            roc_auc = roc_auc_score(label_all, lambda_)
            wandb.log({"Test AUC": roc_auc})

            # show scores distribution with labels
            plt.figure()
            plt.hist(lambda_[label_all == 0], bins=50, alpha=0.5, label='Class 0')
            plt.hist(lambda_[label_all == 1], bins=50, alpha=0.5, label='Class 1')
            plt.xlabel('Test Statistic')
            plt.ylabel('Frequency')
            plt.title('Test Statistic Distribution')
            plt.legend()
            wandb.log({
                'Test Statistic Distribution': wandb.Image(plt)
            })
            plt.close()
        
        # best_auc = float('-inf')         # 如果你坚持这样初始化也可以
        # best_epoch = None       # 关键：初始化，表示还没有“最优”出现
        # save_dir = os.path.join(args.save_model_path, args.srtype)
        # os.makedirs(save_dir, exist_ok=True)
        # to_save = cls_model.module if hasattr(cls_model, "module") else cls_model

        # # ---- 每个 epoch 评估后 ----
        # if roc_auc > best_auc:
        #     best_auc = roc_auc
        #     best_epoch = epoch

        #     torch.save(to_save.state_dict(),
        #     os.path.join(save_dir, f'ema{args.cls_type}_{args.train_type}_{args.kl}.pth'))
        #     np.save(os.path.join(save_dir, f'{args.cls_type}_{args.train_type}_kl{args.kl}_mu_ema.npy'),   mu_ema_)
        #     np.save(os.path.join(save_dir, f'{args.cls_type}_{args.train_type}_kl{args.kl}_s_ema.npy'),    s_ema_)
        #     np.save(os.path.join(save_dir, f'{args.cls_type}_{args.train_type}_kl{args.kl}_Kinv_ema.npy'), Kinv_ema_)
        #     np.save(os.path.join(save_dir, f'{args.cls_type}_{args.train_type}_kl{args.kl}_K_ema.npy'),    K_ema_)

        # # 仅在 25 轮后检查早停，且要确保 best_epoch 已经产生过
        # if (epoch > 25) and (best_epoch is not None) and ((epoch - best_epoch) > 10):
        #     print("Early stopping at epoch ", epoch)
        #     print("Best epoch is ", best_epoch)
        #     break

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

        if epoch <= 25:
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_epoch = epoch
                # Save the best model
                torch.save(cls_model.module.state_dict(), os.path.join(args.save_model_path, args.srtype) + '/ema{}_{}_{}_{}_depth{}_z{}_io{}.pth'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss))
                # save mu_ema, s_ema, Kinv_ema, K_ema
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_mu_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), mu_ema_)
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_s_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), s_ema_)
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_Kinv_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), Kinv_ema_)
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_K_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), K_ema_)


        if epoch > 25:
            if roc_auc > best_auc:
                best_auc = roc_auc
                best_epoch = epoch
                # Save the best model
                torch.save(cls_model.module.state_dict(), os.path.join(args.save_model_path, args.srtype) + '/ema{}_{}_{}_{}_depth{}_z{}_io{}.pth'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss))
                # save mu_ema, s_ema, Kinv_ema, K_ema
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_mu_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), mu_ema_)
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_s_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), s_ema_)
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_Kinv_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), Kinv_ema_)
                np.save(os.path.join(args.save_model_path, args.srtype) + '/{}_{}_kl{}_lr{}_depth{}_z{}_io{}_K_ema.npy'.format(args.cls_type, args.train_type, args.kl, args.lr, args.depth, args.z_dim, args.ioloss), K_ema_)
            else:
                if (epoch - best_epoch) > 10:
                    print("Early stopping at epoch ", epoch)
                    print("Best epoch is ", best_epoch)
                    break
    

if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
        # gpu 
    gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id 
    
    # Remove the first argument (GPU ID) from sys.argv so argparse doesn't see it
    if len(sys.argv) > 1:
        sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--z_dim', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--depth', type=int, default=6, help='1-10')
    parser.add_argument('--lpips', type=bool, default=False)
    parser.add_argument('--cycoptim', type=bool, default=False)
    parser.add_argument('--cls_type', type=str, default='VIBCNN', help='CNN, ResNet, HO, VIBHO')
    parser.add_argument('--kl', type=float, default=500, help='0.1 for VIBHO, 0.0005 for VIBCNN')
    parser.add_argument('--ioloss', type=float, default=500, help='0.1 for VIBHO, 0.0005 for VIBCNN')

    # Paths and settings
    parser.add_argument('--srtype', type=str, default='UNet', help='UNet, UNet_Small, UNet_Tiny')
    parser.add_argument('--proporation', type=float, default=1.0, help='0.001, 0.01, 0.1, 0.2, 0.5, 1.0')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--train_type', type=str, default='cls', help='cls, measure, recon')

    parser.add_argument('--data', type=str, default='sks_3_0.04_15_c2_num_signals_noshuffle')
    parser.add_argument('--train_image_path', type=str, default=None)
    parser.add_argument('--test_image_path', type=str, default=None)
    parser.add_argument('--val_image_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default=None)

    args = parser.parse_args()

    
    args.save_model_path = 'checkpoint/{}/{}'.format(args.data, args.proporation)   
    args.train_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/train/data.h5'.format(args.data)
    args.test_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/test/data.h5'.format(args.data)
    args.val_image_path = '/shared/anastasio-s2/SI/HCP_selected/{}/val/data.h5'.format(args.data)
    
    train(args)
