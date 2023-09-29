#!/bin/bash

while getopts a:b:c:d:e:f:g:h:i:j flag
do
    case "${flag}" in
        a) root=${OPTARG};;
        b) conda=${OPTARG};;
        c) config=${OPTARG};;
        d) out=${OPTARG};;
        e) txt=${OPTARG};;
        f) seeds=${OPTARG};;
        g) model_settings=${OPTARG};;
        h) name=${OPTARG};;
        i) adaptations=${OPTARG};;
        j) save_txt=${OPTARG};;
    esac
done
                
cd $root

source /itet-stor/begressy/net_scratch/conda/bin/activate $conda
python main.py --config_path $config --log_folder_name $out $txt --features raw --num_seeds $seeds --model_settings $model_settings --unique_name $name $adaptations $save_txt