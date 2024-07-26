yaml_file=fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes
subroot=seg
data_path=$1
if [ ! -d "config/mm${subroot}/${yaml_file}" ];then
    mkdir -p config/mm${subroot}/${yaml_file}
    mim download mmsegmentation --config ${yaml_file} --dest config/mm${subroot}/${yaml_file}/
    ckpt_file=\'$(ls config/mm${subroot}/${yaml_file}/*.pth)\'
    config_file=$(ls config/mm${subroot}/${yaml_file}/*.py)
    # modify ckpt file path and data root path in config file here
    sed -i "s#load_from = None#load_from = $ckpt_file#g" $config_file
    sed -i "s#data/#$data_path#g" $config_file
fi