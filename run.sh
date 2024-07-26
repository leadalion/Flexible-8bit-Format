python main.py --dataset ImageNet --net efficientNet_b3 --ckpt ckpt/teacher/spring_regnetx_600m.pth --data_path data/imagenet \
--logger --batch_size 128 --gpu_id 2 --quant_all --cali_num 256 --fold_bn \
--w_scale_method max --a_scale_method max --wbit 8 --abit 8  \
--wfmt stpu --afmt stpu --enable_int --cali_mode batch --skip_fptest --mo4w 