nohup python train.py \
        --unify --switch --moe --amp \
        --name omni-switch-moe-xs-layer1-expert4-k2-mlp4 \
        --model_name rwkv7_hira_switch_moe --model_size xs \
        --pretrain_model checkpoints/omnicomp/omni_xs_layer1_expert4_k2_mlp4.pth \
        --gpu_ids 2 --batch_size 64 --nepochs 16 \
        > ./logs/omni-switch-moe-xs-layer1-expert4-k2-mlp4.log