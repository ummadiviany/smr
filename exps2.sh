# /home/amritesh/anaconda3/envs/torch/bin/python train_joint.py  --device cuda:0 --epochs 80 --lr 1e-3 --seed 2000 --wandb_log
# /home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --l2reg --alpha 0.1 --order prostate158,isbi,promise12,decathlon --device cuda:1 --optimizer adam --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename prostate_l2reg --wandb_log
# /home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --latent_replay --store_samples 3 --order prostate158,isbi,promise12,decathlon --device cuda:1 --optimizer adam --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename prostate_latent --wandb_log
# /home/amritesh/anaconda3/envs/torch/bin/python train_joint.py --device cuda:1 --epochs 80 --lr 1e-3 --seed 2000 --filename hippo_joint --roi_size 128 --wandb_log
# /home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --order harp,drayd --optimizer sgd --device cuda:1 --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename hippo_seq --roi_size 128 --wandb_log
# /home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --l2reg --alpha 0.1 --order harp,drayd --optimizer adam --device cuda:1 --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename hippo_l2reg --roi_size 128 --wandb_log
/home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --ewc --ewc_weight 1.0 --order harp,drayd --optimizer adam --device cuda:1 --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename hippo_ewc --roi_size 128 --wandb_log
/home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --latent_replay --store_samples 3 --order harp,drayd --optimizer adam --device cuda:1 --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename hippo_latent --roi_size 128 --wandb_log
/home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --replay --store_samples 2 --sampling_strategy random --order harp,drayd --optimizer adam --device cuda:1 --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename hippo_replay3 --roi_size 128 --wandb_log
/home/amritesh/anaconda3/envs/torch/bin/python train_seq.py --replay --cropstore --store_samples 9 --sampling_strategy gpcc --order harp,drayd --optimizer adam --device cuda:1 --initial_epochs 80 --lr 1e-3 --lr_decay 0.8 --epoch_decay 0.8 --seed 2000 --filename hippo_gpcc --roi_size 128 --wandb_log
