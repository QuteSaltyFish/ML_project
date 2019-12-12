import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import multiprocessing

import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", type=str, help="The data root")
    parser.add_argument("--origin", type=int,
                        help="The data to be argumented")
    parser.add_argument("--idx", type=int,
                        help="The now index")
    # parser.add_argument("--name", default='VoxNet_v1_{}'.format(LR), type=str,
    #                     help="Whether to test after training")
    args = parser.parse_args()

    data_root = args.dir 
    origin_name = args.origin
    idx = args.idx
    [voxel, seg] = t.load(os.path.join(data_root, '{}.pt'.format(origin_name)))
    # use permute to change the point of view
    voxeltmp = voxel.permute([0, 2, 1])
    segtmp = seg.permute([0, 2, 1])
    t.save([voxeltmp, segtmp], os.path.join(data_root, '{}.pt'.format(idx)))
    idx += 1

    # mirror the image
    # x axis
    voxeltmp =  voxel.flip(-1)
    segtmp =  seg.flip(-1)
    t.save([voxeltmp, segtmp], os.path.join(data_root, '{}.pt'.format(idx)))
    idx += 1
    # y axis
    voxeltmp =  voxel.flip(-2)
    segtmp =  seg.flip(-2)
    t.save([voxeltmp, segtmp], os.path.join(data_root, '{}.pt'.format(idx)))
    idx+=1
    # z axis
    voxeltmp =  voxel.flip(-3)
    segtmp =  seg.flip(-3)
    t.save([voxeltmp, segtmp], os.path.join(data_root, '{}.pt'.format(idx)))
    idx+=1
    print('{}th data argumentation done.'.format(origin_name))
    