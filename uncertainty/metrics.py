import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
import cv2
import os 
import pdb
import matplotlib
import matplotlib.pyplot as plt

## this probably not the best way to do this
softmax_ = nn.Softmax(dim=1)


class Umetrics:

    def __init__(self, no_classes, pred_scores, labels_pred, images, labels, save_id, compute_metrics = False, save_metrics = False, save_softmax_only = False, meta_save_path=None):

        self.image = images
        self.no_classes = no_classes
        self.labels_pred = labels_pred  # numpy array

        self.pred_scores = pred_scores.cpu().numpy()
        self.softmax_map = softmax_(pred_scores).cpu().numpy()
        self.softmax_map_max = np.max(self.softmax_map, axis = 1)
        self.labels = labels.cpu().numpy()
        self.save_id = save_id
        self.meta_save_path = meta_save_path
        if compute_metrics:
            self.compute_umetrics()


        if save_metrics:

            if save_softmax_only:
                self.save_softmax_image()
            else:
                self.save_metrics_image()
           

    def compute_umetrics(self):
        self.entropy_map = self.compute_entropy()
        self.prob_margin_map = self.compute_probability_margin()
        self.var_ratio_map = self.compute_variation_ratio()

     
    def save_metrics_image(self):
        for i in range(len(self.labels_pred)):
            save_path = self.meta_save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(save_path + 'sample_batch{}_image{}'.format(self.save_id, i), pred = self.labels_pred[i,:], label = self.labels[i,:], scores = self.pred_scores[i,:], entropy = self.entropy_map[i,:], prob_margin = self.prob_margin_map[i,:], var_ratio = self.var_ratio_map[i,:])

    def save_softmax_image(self):
        for i in range(len(self.softmax_map)):
            save_path = self.meta_save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(save_path + 'sample_batch{}_image{}'.format(self.save_id, i), softmax_map = self.softmax_map[i,:], label = self.labels[i,:])

    def compute_entropy(self):
        entropy_map = entropy(self.softmax_map, axis = 1) #/np.log(19)
        return entropy_map


    def compute_probability_margin(self):
        softmax_map_partitioned = np.partition(self.softmax_map, self.no_classes-2, axis = 1)
        prob_margin_map = 1 - softmax_map_partitioned[:,self.no_classes-1, :] + softmax_map_partitioned[:, self.no_classes-2,:]
        return prob_margin_map


    def compute_variation_ratio(self):
        var_ratio_map = 1 - np.max(self.softmax_map, axis = 1)
        return var_ratio_map
