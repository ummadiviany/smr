# /home/amritesh/anaconda3/envs/torch/bin/python train_joint.py  --device cuda:0 --epochs 80 --lr 1e-3 --seed 2000 --wandb_log
/home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --l2reg --alpha 0.1 --order prostate158,isbi,promise12,decathlon --device cuda:1 --optimizer adam --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename prostate_l2reg 
