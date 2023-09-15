import numpy  as np
import os
import argparse
import logging
import json
import time
import random
import sys 

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import jaccard_score, confusion_matrix
import numpy.ma as ma
from torch.nn import functional as F

from dataloader import CustomDataset
from autoencoder_skip import UNet

from ap_metrics import APMetrics
import matplotlib
import matplotlib.pyplot as plt

import pdb
sigmoid  = nn.Sigmoid()
relu = nn.ReLU()


def get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type = str, default = '../datasets/data/')
    parser.add_argument("--folder_tag", type = str, default = 'meta_data')
    parser.add_argument("--ood_tag", type = str, default = 'fs_val')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("--ckpt", type = str, default = None)
    parser.add_argument("--save_path", type = str, default = None)
    parser.add_argument("--single_model", action='store_true', default = False)

    parser.add_argument("--train", action = 'store_true', default = False)
    parser.add_argument("--eval", action = 'store_true', default = False)
    parser.add_argument("--continue_training", action = 'store_true', default = False)
    parser.add_argument("--test_only", action = 'store_true', default = False)
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--lr", type = float, default = 0.01)
    parser.add_argument("--optim", type = str, default = 'Adam', choices=['Adam', 'SGD'])
    parser.add_argument("--loss", type = str, default = 'BCE', choices=['BCE', 'MSE']) 
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                            help='weight decay (default: 1e-4)')

    parser.add_argument("--total_epochs", type = int, default = 50)
    parser.add_argument("--no_channels", type = int, default = 4)
    parser.add_argument("--random_seed", type = int, default = 3)
    parser.add_argument("--file", type=str, default='out.txt')
    
    return parser

def get_dataset(data_root, folder_tag,ood_tag,no_channels):


    # mean = min_values
    # std = np.subtract(max_values,min_values)
    mean = [0.0, 0.0]
    std = [2.864, 18]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(tuple(mean),tuple(std))])
    # transform = transforms.Compose([transforms.ToTensor()])
    
    data_path_id_train = data_root+ folder_tag +'/train/'
    data_path_id_val = data_root+folder_tag +'/val/'
    data_path_ood = os.path.join(data_root,folder_tag,ood_tag) + '/'
   
    print(data_path_id_train, data_path_id_val, data_path_ood)
    ood_set = CustomDataset(data_path = data_path_ood, split = 'ood', transform = transform, no_channels = no_channels)
    trainset = CustomDataset(data_path = data_path_id_train, split = 'id', transform = transform, no_channels = no_channels)
    valset = CustomDataset(data_path = data_path_id_val, split = 'id', transform = transform, no_channels = no_channels)  
    
    print("Train length: %d" % len(trainset))
    print("Val length: %d" % len(valset))
    print("OOD length: %d" % len(ood_set))
    return trainset, valset, ood_set


def save_ckpt(path, curr_iters, model_state, optimizer_state, score):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)


def visualize(pred, label,ent,img_id):

    fig, axis = plt.subplots(1,3)
    axis[0].imshow(pred[0,:,:])
    axis[1].imshow(ent[0,:,:])
    axis[2].imshow(label[0,:,:])

    #plt.savefig('results_useg/model15image_%d' % img_id, bbox_inches='tight', pad_inches=0)
    #plt.close()
    plt.show()

def visualize_val(umask_soft, img_id):
    ## do this with a batch size of 1     
    pred = (umask_soft[0]*255).astype(np.uint8)/255.0

    plt.imshow(pred, cmap='viridis', interpolation ='nearest')
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    plt.savefig('results_fs/%d_pred_masked.png'%img_id, dpi = 300)


def validate(data, model, device, metrics, ood = False, baseline = False):
    baseline = False
    model.eval()
    if baseline:
        metrics_ent = APMetrics(2)
    metrics.reset()
    score = 0
    img_id = 0
    y_true_full = []
    y_pred_full = []
    y_pred_ent_full = []
    with torch.no_grad():
        for images, labels, masks, labels_id_true, labels_id_pred, ent in data:

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.int)
            masks = masks.to(device)
            labels_id_true = labels_id_true.to(device, dtype=torch.int)
            labels_id_pred = labels_id_pred.to(device, dtype=torch.int)
            umask_soft = torch.squeeze(model(images),1)

            umask_soft = sigmoid(umask_soft)
            umask = torch.where(umask_soft < 0.9, 0, 1)

            labels = labels.cpu().numpy()
            masks = masks.cpu().numpy()

            img_id +=1
            if not ood:
                labels = 1-labels 
                umask_soft = 1-umask_soft

            labels_id_true = labels
            y_true = ma.masked_array(labels_id_true, masks)
            y_true = y_true.data[y_true.mask]

            if baseline:
                y_pred_ent = ma.masked_array(ent, masks)
                y_pred_ent = y_pred_ent.data[y_pred_ent.mask]
            
            
            if np.count_nonzero(y_true) == 0 :
                #print("skipping this data point")
                continue
            # if len(y_true)==0 or len(ent)==0:
            #     continue

            umask_soft = umask_soft.cpu().numpy()

            ## remove this 
            
            y_pred = ma.masked_array(umask_soft, masks)
            y_pred = y_pred.data[y_pred.mask]

            

            # visualize(umask_soft*masks,labels_id_true*masks, ent*masks, img_id)
            # visualize_val(umask_soft*masks, img_id)
            y_pred_full += list(y_pred)
            y_true_full += list(y_true)
            if baseline:
                y_pred_ent_full += list(y_pred_ent)

                # # metrics.update(y_true, y_pred)
                # metrics_ent.update(y_true, y_pred_ent)
                # # print("BATCH ID", img_id)
                # # print(metrics.get_results())
                # print(metrics_ent.get_results())

    metrics.update(y_true_full, y_pred_full)
    fpr95, tpr = metrics.compute_fpr(y_true_full, y_pred_full)
    if baseline:
        metrics_ent.update(y_true_full, y_pred_ent_full)    
        fpr95_bl, tpr_bl = metrics_ent.compute_fpr(y_true_full, y_pred_ent_full)
    print("ours :", metrics.get_results())
    print("FPR-95 :", fpr95, tpr)
 
    if baseline:
        print("entropy baseline :", metrics_ent.get_results())
        print("FPR-95 baseline :", fpr95_bl, tpr_bl)
    score = metrics.get_results()
    return score

def evaluate(model, val_data, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels, loss_mask, _ , _, _ in val_data:

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            loss_mask = loss_mask.to(device)

            output = model(images)
            output = torch.squeeze(output, 1)
            loss_full = criterion(output, labels)
            loss = (loss_full*loss_mask.float()).sum()
            nonzero_num = loss_mask.sum()
            loss = loss/nonzero_num
            val_loss += loss

        val_loss = val_loss/len(val_data)
        return val_loss

def train(train_data, val_data, model, optimizer, criterion, device, metrics, total_epochs, curr_epoch, save_path):
    ### incomplete

    for epoch in range(curr_epoch,total_epochs):
        epoch_loss = 0
        prev_loss = 1e7
        model.train()
        start_time = time.time()
        for images, labels, loss_mask, _ , _, _ in train_data:

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            loss_mask = loss_mask.to(device)

            optimizer.zero_grad()
            output = model(images)
            output = torch.squeeze(output, 1)
            loss_full = criterion(output, labels)
            #loss = loss_full

            loss = (loss_full*loss_mask.float()).sum()
            nonzero_num = loss_mask.sum()
            loss = loss/nonzero_num

            loss.backward()
            optimizer.step()
            # print(loss)
            epoch_loss += loss
        epoch_loss = epoch_loss/len(train_data)
        end_time = time.time()
        print(f'\t EPOCH: {epoch+1:.0f} | Train Loss :{epoch_loss:.3f}')
        # print(f'\t Time elapsed: {end_time - start_time:.3f}')

        ## validation 

        val_loss = evaluate(model, val_data, criterion, device)
        print("Val set loss :",val_loss)

        if epoch_loss < prev_loss:
            #prev_loss = epoch_loss
            torch.save({"cur_itrs": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "score": epoch_loss}, save_path+'/model_{}.pt'.format(epoch))

        

def main():

    opts = get_argparser().parse_args()
    # orig_stdout = sys.stdout 
    # f = open(opts.file, 'w')
    # sys.stdout = f 

    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.backends.cudnn.deterministic = True


    if opts.train:
        if not os.path.exists(opts.save_path):
            os.makedirs(opts.save_path)
        with open(opts.save_path+'/args.txt', 'w' ) as f:
            json.dump(opts.__dict__, f, indent=2)

    ## set gpu id
    # os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    ## TODO: setup manual_seed, does the segmenation random seed matter

    train_set,val_set, ood_set = get_dataset(opts.data_root, opts.folder_tag,opts.ood_tag, opts.no_channels)
    if opts.train:
        valloader = data.DataLoader(val_set, batch_size = opts.batch_size , shuffle = False, num_workers = 2)
        trainloader = data.DataLoader(train_set, batch_size = opts.batch_size, shuffle = True, num_workers = 2)
    if opts.eval:
        # oodloader = data.DataLoader(val_set, batch_size = 1, shuffle = False, num_workers = 8)
        oodloader = data.DataLoader(ood_set, batch_size = 1, shuffle = False)

    ## load model
    
    model = UNet(in_channels = opts.no_channels, input_shape = (512, 1024))
    
    model.to(device)
    if opts.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = opts.lr)
    elif opts.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = opts.lr, momentum = 0.9, weight_decay = opts.weight_decay)

    else:
        print("Inavlid optimizer: %s", opts.optim)

    if opts.loss == 'BCE':
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')
    elif opts.loss == "MSE":
        criterion = nn.MSELoss(reduction = 'none')
    else:
        print("Invalid loss type %s", opts.loss_type)

   
    metrics = APMetrics(2)

    def load_model(ckpt_path, model):

        if ckpt_path is not None and os.path.isfile(ckpt_path):

            checkpoint = torch.load(ckpt_path, map_location = device)
            model.load_state_dict(checkpoint["model_state"])
            # model = nn.DataParallel(model) figure this out
            model.to(device)
        else:
            print("Check specified model path")
            exit(0)

    if opts.eval:

        # model.eval()
        print("OOD SET")
        if opts.single_model:
            load_model(opts.ckpt, model)
            model.eval()
            ood_score = validate(oodloader, model, device, metrics, ood = True, baseline = True)
            print(ood_score)
        else:
            for e in range(0, opts.total_epochs):
                print("Model :", e)
                ckpt_path = os.path.join(opts.ckpt, 'model_{}.pt'.format(e))
                load_model(ckpt_path, model)
                model.eval()
                ood_score = validate(oodloader, model, device, metrics, ood = True, baseline = True)
                print(ood_score)
               
        return

    elif opts.continue_training:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        #scheduler
        cur_itrs = checkpoint["cur_itrs"]
        score = checkpoint['score']
        print("Training restored from %s " % opts.ckpt)
        print("Model restored from %s " % opts.ckpt)
        del checkpoint
        train(trainloader, valloader, model, optimizer, criterion, device, metrics, opts.total_epochs, cur_itrs, opts.ckpt)

    else:
        print("Training from scratch")
        train(trainloader, valloader, model, optimizer, criterion, device, metrics, opts.total_epochs, curr_epoch = 0, save_path = opts.save_path)
    
    # sys.stdout = orig_stdout 
    # f.close()
if __name__ == '__main__':

    main()
