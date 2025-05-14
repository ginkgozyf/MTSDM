epoch=70
lr=2e-3
dataset_=('FD001' 'FD002' 'FD003' 'FD004')
arch_=('original') # 'att' 'original' 'att+td' 'td'
window_size_=(24 48 96)
state='eval' # train,sample,eval
schedule_name='linear' #linear, cosine
loss_type='mse' #mse mse+mmd
T=500

for dataset in "${dataset_[@]}"
do
    for window_size in "${window_size_[@]}"
    do
        for arch in "${arch_[@]}"
        do
            CUDA_VISIBLE_DEVICES=0 python MainCondition.py --lr=${lr} \
            --epoch=${epoch} --arch=${arch} --arch=${arch}  --dataset=${dataset}  \
            --window_size=${window_size}  --state=${state}  \
            --schedule_name=${schedule_name} --loss_type=${loss_type} \
            --T=${T}
        done
    done
done
