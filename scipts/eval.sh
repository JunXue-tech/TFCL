export PATH='your conda envs path'

DEVICE=cuda:1
batch_size=16
# change the model you wanna test
MODEL_PATH="your model path"
OUT_PATH="your output path"
code='./TFCL/code/eval.py'
#dataset 
#ASVspoof2019la 
LA19_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_eval.txt'
LA19_wav_path='./dataset/ASVspoof2019/ASVspoof2019_LA_eval'

#ASVspoof2019la sim echo
LA19_echo_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_echo_eval.txt'
LA19_echo_wav_path='./dataset/ASVspoof2019/ASVspoof2019_echo_eval'
#ASVspoof2019la webrtc aec
LA19_aec_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_aec_eval.txt'
LA19_aec_wav_path='./dataset/ASVspoof2019/ASVspoof2019_aec_eval'

#ASVspoof2019la add noise
LA19_noisy_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_noisy_eval.txt'
LA19_noisy_wav_path='./dataset/ASVspoof2019/ASVspoof2019_noisy_eval'

#ASVspoof2019la webrtc
LA19_ns_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_ns_eval.txt'
LA19_ns_wav_path='./dataset/ASVspoof2019/ASVspoof2019_ns_eval'

#ASVspoof2019la webrtc agc
LA19_agc_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_agc_eval.txt'
LA19_agc_wav_path='./dataset/ASVspoof2019/ASVspoof2019_agc_eval'

#ASVspoof2019la webrtc vad
LA19_vad_protocol_path='./dataset/ASVspoof2019/ASVspoof2019_vad_eval.txt'
LA19_vad_wav_path='./dataset/ASVspoof2019/ASVspoof2019_vad_eval'


echo ASVspoof2019LA
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA \
  --protocol_path $LA19_protocol_path \
  --wav_path  $LA19_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE


echo ASVspoof2019LA echo
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA_echo \
  --protocol_path $LA19_echo_protocol_path \
  --wav_path  $LA19_echo_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE


echo ASVspoof2019LA aec
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA_aec \
  --protocol_path $LA19_aec_protocol_path \
  --wav_path  $LA19_aec_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE

echo ASVspoof2019LA noisy
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA_noisy \
  --protocol_path $LA19_noisy_protocol_path \
  --wav_path  $LA19_noisy_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE

echo ASVspoof2019LA ns
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA_ns \
  --protocol_path $LA19_ns_protocol_path \
  --wav_path  $LA19_ns_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE
  
echo ASVspoof2019LA agc
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA_agc \
  --protocol_path $LA19_agc_protocol_path \
  --wav_path  $LA19_agc_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE
  

echo ASVspoof2019LA vad
python $code \
  --model_path $MODEL_PATH \
  --dataset ASVspoof2019LA_vad \
  --protocol_path $LA19_vad_protocol_path \
  --wav_path  $LA19_vad_wav_path \
  --output_dir $OUT_PATH \
  --batch_size $batch_size \
  --num_workers 4 \
  --device $DEVICE