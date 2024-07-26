
# # All Mixed quant batch
Task="rte"
python main_glue.py --task $Task \
--logger --batch_size 8 --gpu_id 0 --quant_all --cali_num 256 --fold_bn \
--w_scale_method mse --a_scale_method max --wbit 4 --abit 8  \
--wfmt stpu_extend --afmt stpu --enable_int --cali_mode batch \
--ckpt bert-base-uncased-$Task/pytorch_model.bin
