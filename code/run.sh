# settings.
gpu=3
seed=321

epochs_warmup=10
train_data_percent=1
cluster_num=800
bs=128
python run_train_base.py --seed $seed --net_nums dual_networks --training_mode 3.0 --gpu $gpu --net pre_resnet18 \
 --lr 0.02 -bs $bs --epochs 300 --epochs_warmup $epochs_warmup --cluster_interval 10 --cluster_num $cluster_num \
 --noise_mode worse_label --noise_rate -1 --train_data_percent $train_data_percent
