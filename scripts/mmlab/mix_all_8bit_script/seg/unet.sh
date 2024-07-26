yaml_file=fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes
TIME=$(date +"%Y%m%d_%H%M%S")
subroot=seg
task=log_mix8
if [ ! -d "log/${task}/${subroot}" ];then
    mkdir -p log/${task}/${subroot}
fi
python main.py --dataset mm${subroot} --model_zoo mm${subroot} \
--net config/mm${subroot}/${yaml_file}/${yaml_file}.py \
--logger \
--wbit 8 --abit 8 \
--enable_int \
--mo4w \
--cali_num 256 --fold_bn --w_scale_method max --a_scale_method max \
--wfmt stpu --afmt stpu --wnorm 2 --anorm 2 2>&1 | tee log/${task}/${subroot}/out-${yaml_file}-${TIME}.log
