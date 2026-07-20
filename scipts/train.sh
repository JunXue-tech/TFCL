export PATH='your conda envs path'

save_model='your path'

python ./TFCL/code/main_train.py \
    --track ASVspoof2019_afe \
    --batch_size 4 \
    --lr_det 1e-6 \
    --max_epochs 100 \
    --patience 10 \
    --train_data_path your_train_data_path \
    --dev_data_path your_dev_data_path \
    --protocols_path your_protocols_path \
    --clean_train_data_path your_clean_train_data_path \
    --clean_dev_data_path your_clean_dev_data_path \
    --clean_protocols_path your_clean_protocols_path \
    --proc_suffix _echoaec_noisyns_agc_vad \
    --align_weight 0.3 \
    --out_path $save_model \
    --device cuda:1 \
    --num_workers 8 \
    --train_tag TFCL