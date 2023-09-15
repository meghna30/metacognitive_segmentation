import numpy as np 
import os 
import argparse
import numpy.ma as ma
import pdb 
from ap_metrics import APMetrics

def get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type = str, default = '/mnt/kostas-graid/datasets/meghnag/data/baselines')
    parser.add_argument("--baseline", type=str, default='ens2')
    parser.add_argument("--num_models", type=int, default = 3)
    parser.add_argument("--ood_dataset", type=str, default='fs_val')

    return parser


def compute_ood_scores(sfm_maps):
    """
    num_models*num_classes*h*w
    """
    var = np.var(sfm_maps, axis = 0)
    # pdb.set_trace()
    ood_scores = np.sum(var, axis = 0)

    return ood_scores

def main():
    
    opts = get_argparser().parse_args()

    data_paths = []

    ## get the model data paths
    metrics = APMetrics(2)
    for i in range(0, opts.num_models):

        curr_path = os.path.join(opts.data_root, opts.baseline, 'model{}'.format(i+1),  opts.ood_dataset)
        data_paths.append(curr_path)

    file_names = os.listdir(data_paths[0])

    # sfm_maps  = []
    # masks = []
    y_pred_full = []
    y_true_full = []
    i = 0
    for f in file_names:
        #print(i)
        for j in range(0, opts.num_models):
            curr_sample = np.load(os.path.join(data_paths[j], f))
            curr_map = np.expand_dims(curr_sample['softmax_map'], axis = 0)
            if j == 0:
                sfs_maps = curr_map
            else:
               
                sfs_maps = np.vstack((sfs_maps, curr_map)) 
           
        labels = curr_sample['label']
        mask = np.where(labels != 255, 1, 0) 
        labels = np.where(labels == 2, 1, 0)
        
        ood_scores = compute_ood_scores(sfs_maps)
       
        y_pred = ma.masked_array(ood_scores, mask)
        y_pred = y_pred.data[y_pred.mask]

        y_true = ma.masked_array(labels, mask)
        y_true = y_true.data[y_true.mask]

        # if np.count_nonzero(y_true) != 0 :
        y_pred_full += list(y_pred)
        y_true_full += list(y_true)
        i+=1
        
    metrics.update(y_true_full, y_pred_full)
    fpr95, tpr = metrics.compute_fpr(y_true_full, y_pred_full)
    print("FPR-95 :", fpr95, tpr)
    print("ours :", metrics.get_results())

if __name__=='__main__':
    main()    

