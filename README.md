Install requirements.yaml to setup the python environment 

# Implementation Details 

1. Download Citysapes, Lost and Found datasets and the Fishyscapes Beanchmark. 

2. To train the MaxEnt Model 

`python main.py --model deeplabv3plus_resnet101 --dataset cityscapes --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --model_tag maxent_model --baseline --ood_train_data --entropy_reg --sigma 0.06`

 The models gets stored in the checkpoints folder, make sure to set correct data paths

 3. To generate the training data for the metacognitive model

    train set :

`python main.py --model deeplabv3plus_resnet101 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --test_only --ckpt checkpoints/maxent_model.pt --baseline --save_umetrics --meta_save_path ./meta_data/train/ --meta_train_data 
`
   Val set 

`python main.py --model deeplabv3plus_resnet101 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 4 --output_stride 16 --data_root /data/cityscapes --test_only --ckpt checkpoints/maxent_model.pt --baseline --save_umetrics --meta_save_path /meta_data/val/
`
 4. To train the metacognition network, navigate to the uncertainty folder 

` python meta_main.py --save_path metacog --train --no_channels 2 --batch_size 4 --folder_tag meta_data `
 
 make sure the that the folder tag and data paths in meta_main.py point to the right directories

5. To generate the OOD test data for the metacognitive model

`python main.py --model deeplabv3plus_resnet101 \
 --gpu_id 0  --lr 0.1  --crop_size 768 --batch_size 4 --output_stride 16 \
 --data_root /data/cityscapes --test_only \
 --ckpt checkpoints/maxent_model.pt \
 --baseline --ood_data --ood_seg --ood_dataset fishyscapes --ood_data_root /data/fishyscapes/
 --save_umetrics --_meta_save_path /meta_data/fs_val/ 
`
 6. To evaluate the MaxEnt + metacognition network - navigate to the uncertainty folder

` python meta_main.py --ckpt metacog/model.pt --eval --no_channels 2 --batch_size 1 --folder_tag meta_data --ood_tag fs_val --single_model 
`
for fishyscapes datsets uncomment line 50 in uncertainty/dataloder.py


# Training Details 


1. MaxEnt model is learnt by fine-tuning the base segmentation model using the synthetically generated OOD data set and entropy regularization.  We using a SGD optmizer with an initial learning rate if 0.01 and weight decay of 1e − 4 and use poly learning rate scheduler, with a power of 0.9 over the total training epochs. The entropy regularization parmater λ = 0.06.

2. Metacognitive network : We train the Unet with a learning rate to 0.01 and optimize for the binary cross entropy loss using Adam optimization.

3. For generating the synthetic OOD samples we use randomly sampled subset D sub and C sub from D_id . For Gaussian blurring the pixels in D sub that do not belong to C sub we use a kernel size of 35 and a standard deviation of 20. Sample images are in folder synthOOD




