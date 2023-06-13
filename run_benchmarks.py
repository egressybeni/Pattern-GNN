import os
import subprocess
import time

### Batch Job args ###
conda_environment_path = 'oko'
n_gpus = 1
memory_gb = 60
######################

explore = False

DATA = 'SJ'
NUM = '99' if explore else '06'
out_folder = f'../logs/logs_{DATA}_{NUM}'
num_seeds = 1 if explore else 5
model_settings = f'./model_settings/model_settings_{DATA}.json'
save_model = False

### Simulator ###
# sim_labels      = 'binary_all'
sim_labels      = 'binary_complex'
sim_generator   = 'chordal'
sim_num_nodes   = 8192
sim_avg_degree  = 6
#################

if DATA == 'ETH':
    data_txt = '--readout node --simple_efeats'
elif DATA == 'SIM':
    data_txt = f'--readout node --graph_simulator --y_pretrain {sim_labels} --sim_num_nodes {sim_num_nodes} --sim_generator {sim_generator} --sim_avg_degree {sim_avg_degree}'
else:
    data_txt = '--readout edge'

save_model_txt = '--save_model' if save_model else ''

config_template = "./configs/{accr}_{data}_explore.json" if explore else "./configs/{accr}_{data}.json"
unique_template = "{data}_{num}_{accr}"

runs = {
    'gin': [
        # '--reverse_mp --ports --ego --disjoint --edge_updates2 ',
        # '--reverse_mp --ports --ego --disjoint ',
        # # # '--reverse_mp --ports --edge_updates2 ',
        # '--reverse_mp --ports ',
        # '--reverse_mp ',
        # '',
        # '--edge_updates2 ',
        '--ports',
        '--ego --disjoint'
    ],
    'pna': [
        # '--reverse_mp --ports --ego --disjoint --edge_updates2 ',
        # '--reverse_mp --ports --ego --disjoint ',
        # # '--reverse_mp --ports --edge_updates2 ',
        # # '--reverse_mp --ports ',
        # # '--reverse_mp ',
        # '',
    ],
    'gat': [
        # '--reverse_mp --ports --ego --disjoint --edge_updates2 ',
        # '--reverse_mp --ports --ego --disjoint ',
        # '--reverse_mp --ports ',
        # '--reverse_mp ',
        # '',
    ],
    'mlp': [
        # '',
    ]
}

if __name__ == "__main__":
    os.chdir(os.path.abspath(''))


    for gnn_accr, adaptations_list in runs.items():
        config = config_template.format(accr=gnn_accr, data=DATA)
        unique_name = unique_template.format(accr=gnn_accr, data=DATA, num=NUM)
        # gpu_type = '-require a100_80gb ' if require_a100(data=DATA, gnn_accr=gnn_accr) else ''

        if adaptations_list:
            for adaptations in adaptations_list:
                # if 'ego' in adaptations:
                #     if DATA in ['SJ', 'MK']:
                #         config_run = config[:-5] + '_ego.json'
                # else:
                config_run = config
                # gpu_type = 'geforce_rtx_2080_ti|titan_rtx|geforce_rtx_3090' if 'ego' in adaptations else 'tesla_v100|geforce_rtx_2080_ti|titan_rtx'
                gpu_type = 'geforce_rtx_3090|rtx_a6000' if 'reverse_mp' in adaptations else 'tesla_v100|geforce_rtx_2080_ti|titan_rtx'
                # gpu_type = 'geforce_rtx_3090'
                # if explore:
                #     gpu_type = 'geforce_rtx_3090|rtx_a6000'
                gpu_type = 'tesla_v100|geforce_rtx_2080_ti|titan_rtx'
                # gpu_type = 'geforce_rtx_3090|rtx_a6000'
                # gpu_type = 'rtx_a6000'

                bash_script = """
                sbatch --gres=gpu:1 --constraint='%s' --mem=%sG run.sh -a '%s' -b '%s' -c '%s' -d '%s' -e '%s' -f '%s' -g '%s' -h '%s' -i '%s' -j '%s'
                """ % (gpu_type, memory_gb, os.getcwd(), conda_environment_path, config_run, out_folder, data_txt, num_seeds, model_settings, unique_name, adaptations, save_model_txt)
                
                # python_script = """
                # python main.py --config_path %s --log_folder_name %s %s --features raw --num_seeds %s --model_settings %s --unique_name %s %s %s
                # """% (config_run, out_folder, data_txt, num_seeds, model_settings, unique_name, adaptations, save_model_txt)
                # print(python_script)

                res = subprocess.run(bash_script, capture_output=True, shell=True)
                print(res.stdout.decode())
                time.sleep(1)