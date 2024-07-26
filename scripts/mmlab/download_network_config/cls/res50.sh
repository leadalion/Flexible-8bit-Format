yaml_file=resnet50_8xb32_in1k
subroot=cls
data_path=$1
if [ ! -d "config/mm${subroot}/${yaml_file}" ];then
    mkdir -p config/mm${subroot}/${yaml_file}
    # modify ckpt file path and data root path in config file here
fi
mim download mmcls --config ${yaml_file} --dest config/mm${subroot}/${yaml_file}/
ckpt_file=\'$(ls config/mm${subroot}/${yaml_file}/*.pth)\'
config_file=$(ls config/mm${subroot}/${yaml_file}/*.py)
sed -i "s#load_from = None#load_from = $ckpt_file#g" $config_file
sed -i "s#data/#$data_path#g" $config_file