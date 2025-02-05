import numpy as np
import argparse
import os
import sys
import shutil

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from lib.torch import *
from lib.config import Config
from model.model_lib import model_dict
from lib.utils import prepare_seed, print_log, mkdir_if_missing
import matplotlib.pyplot as plt

def get_model_prediction(data, sample_k):
    model.set_data(data)
    sample_motion_3D, data = model.inference(mode='infer', sample_num=sample_k, need_weights=False)
    sample_motion_3D       = sample_motion_3D.transpose(0, 1).contiguous()
    return sample_motion_3D

def test_model(generator, save_dir, cfg):
    total_num_pred = 0

    # 
    while not generator.is_epoch_end():
        # Get data from the generator (should be a dictionary)
        data = generator()
        if data is None:
            continue

        seq_name, frame = data['seq'], data['frame']
        frame = int(frame)
        sys.stdout.write('testing seq: %s, frame: %06d                \r' % (seq_name, frame))  
        sys.stdout.flush()

        gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * cfg.traj_scale
        with torch.no_grad():
            # 
            sample_motion_3D = get_model_prediction(data, cfg.sample_k)
        sample_motion_3D =  sample_motion_3D * cfg.traj_scale
        sample_motion_3D = sample_motion_3D.cpu().numpy()
        # Plot the prediction and the ground truth
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{seq_name} - frame {frame}')
        for i in range(sample_motion_3D.shape[1]):
            ax.plot(sample_motion_3D[:frame, i, 0], sample_motion_3D[:frame, i, 1], sample_motion_3D[:frame, i, 2], 'b')
            ax.plot(sample_motion_3D[frame:, i, 0], sample_motion_3D[frame:, i, 1], sample_motion_3D[frame:, i, 2], 'r')
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cleanup', action='store_true', default=False)
    args = parser.parse_args()

    # Read the configuration file
    cfg = Config(args.cfg)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if args.gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)
    torch.set_grad_enabled(False)
    log = open(os.path.join(cfg.log_dir, 'log_test.txt'), 'w')


    prepare_seed(cfg.seed)
    # Load model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.eval()
    cp_path = cfg.model_path % 100
    print_log(f'Loading model from checkpoint: {cp_path}', log, display=True)
    model_cp = torch.load(cp_path, map_location='cpu',weights_only=True)
    model.load_state_dict(model_cp['model_dict'], strict=False)

    """ save results and compute metrics """
    generator = data_generator(cfg, log, split='test', phase='testing')
    save_dir = f'{cfg.result_dir}/epoch_{100:04d}/test'; mkdir_if_missing(save_dir)
    eval_dir = f'{save_dir}/samples'
    
    test_model(generator, save_dir, cfg)

    # remove eval folder to save disk space
    if args.cleanup:
        shutil.rmtree(save_dir)


