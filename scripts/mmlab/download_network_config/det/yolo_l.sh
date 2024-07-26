yaml_file=yolox_l_8x8_300e_coco
subroot=det
data_path=$1
if [ ! -d "config/mm${subroot}/${yaml_file}" ];then
    mkdir -p config/mm${subroot}/${yaml_file}
fi
mim download mmdet --config ${yaml_file} --dest config/mm${subroot}/${yaml_file}/
ckpt_file=\'$(ls config/mm${subroot}/${yaml_file}/*.pth)\'
config_file=$(ls config/mm${subroot}/${yaml_file}/*.py)
# modify ckpt file path and data root path in config file here
sed -i "s#data/#$data_path#g" $config_file
sed -i "s#load_from = None#load_from = $ckpt_file#g" $config_file
# fi