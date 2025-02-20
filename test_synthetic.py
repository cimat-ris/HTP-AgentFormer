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


def prepare_data():
    # Define a dictionary to store the data
    simpler_data = {'pre_motion_3D':None, 'fut_motion_3D':None, 'fut_motion_mask':None, 'pre_motion_mask':None, 'pre_data':None, 'fut_data':None, 'heading':None, 'valid_id':None, 'traj_scale':None, 'pred_mask':None, 'scene_map':None, 'seq':None, 'frame':None}

    simpler_data['pre_motion_3D']=[]
    simpler_data['pre_motion_mask']=[]

    for i in range(3):
        simpler_data['pre_motion_3D'].append(torch.zeros(8, 2))
        simpler_data['pre_motion_mask'].append(torch.ones(8, 1))

        # Random initial velocity
        v = torch.rand(1) * 2 - 1
        angle = torch.rand(1) * 2 * np.pi

        # Random initial position
        simpler_data['pre_motion_3D'][i][0,0] = torch.rand(1) * 2 - 1
        simpler_data['pre_motion_3D'][i][0,1] = torch.rand(1) * 2 - 1

        for k in range(1,8):
            simpler_data['pre_motion_3D'][i][k,0] = simpler_data['pre_motion_3D'][i][k-1,0] + v * torch.cos(angle)
            simpler_data['pre_motion_3D'][i][k,1] = simpler_data['pre_motion_3D'][i][k-1,1] + v * torch.sin(angle)
    return simpler_data

def test_model(cfg):

        sys.stdout.write('testing artificial data:  \r')  
        sys.stdout.flush()
        for i in range(10):
            with torch.no_grad():
                # Here is where we simplify the data into a dictionary with just the previous motion information
                mydata = prepare_data() 
                model.set_data(mydata)
                sample_motion_3D, produced_data = model.inference(mode='infer', sample_num=cfg.sample_k, need_weights=False)
                sample_motion_3D                = sample_motion_3D.transpose(0, 1).contiguous()


            sample_motion_3D = sample_motion_3D * cfg.traj_scale
            sample_motion_3D = sample_motion_3D.cpu().numpy()
            pre_motion       = produced_data['pre_motion'].cpu().numpy() * cfg.traj_scale

            # Plot the prediction and the ground truth
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.set_xlim(-1, 3)
            ax.set_ylim(-1, 1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.axis('equal')

            ax.set_title(f'Artificial data ')
            for i in range(1):
                ax.plot(pre_motion[:,i,0], pre_motion[:,i,1], 'bo-')
            #for i in range(1):
            #    ax.plot(gt_motion_3D[i,:,0], gt_motion_3D[i,:,1], 'ro-')
            for i in range(1):
                for k in range(sample_motion_3D.shape[0]):
                    ax.plot(sample_motion_3D[k,i,:,0], sample_motion_3D[k,i,:,1], 'go-')


            plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--epochs', default=None)
    parser.add_argument('--gpu', type=int, default=0)
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
    epoch = 5
    cp_path = cfg.model_path % epoch
    print_log(f'Loading model from checkpoint: {cp_path}', log, display=True)
    model_cp = torch.load(cp_path, map_location='cpu',weights_only=True)
    model.load_state_dict(model_cp['model_dict'], strict=False)

    # Apply model on these data
    test_model(cfg)

