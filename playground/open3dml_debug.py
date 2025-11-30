from pyprojroot import here
import requests
from pathlib import Path
import zipfile
import laspy
import numpy as np

weights_dir = here("data/weights")

import os
import time
import numpy as np
import torch
import torch.nn as nn
import open3d.ml.torch as ml3d
from open3d.ml.torch.datasets import Custom3D
from open3d.ml.torch.modules import losses
from open3d.ml.utils import Config

class CustomPointTransformer(ml3d.models.PointTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_per_class = self.cfg.get('class_weights', None)
        self.class_weights = self.get_class_weights(num_per_class) if num_per_class is not None else None
        self.cce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')

    def get_class_weights(self, num_per_class):
        num_per_class = np.array(num_per_class, dtype=np.float32)
        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return torch.tensor(ce_label_weight)
    
    def get_loss(self, Loss, results, inputs, device):
        labels = inputs['data'].label

        scores, labels = losses.filter_valid_label(
            results, 
            labels, 
            self.cfg.num_classes,
            self.cfg.ignored_label_inds,
            device,
        )
        loss = self.cce_loss(scores, labels).mean()
        return loss, labels, scores
    
model_config = {'name': 'PointTransformer',
 'batcher': 'ConcatBatcher',
 'ckpt_path': str(here("data/weights/model_weights/weights_pointtransformer.pth")),
 'is_resume': False,
 'in_channels': 3,
 'blocks': [2, 3, 4, 6, 3],
 'num_classes': 2,
 'voxel_size': 0.02,
 'max_voxels': 65536,
 'ignored_label_inds': [],
}

pipeline_config = {
'name': 'SemanticSegmentation',
#  'device': 'cuda:0',
'device': 'cpu',
 'num_workers': 0,            # changed: avoid worker subprocesses while debugging
 'batch_size': 1,
}

model = CustomPointTransformer(**model_config)
pipeline = ml3d.pipelines.SemanticSegmentation(model, **pipeline_config)

def read_cloud_old(path):
    if path[-3:] == 'npy':
        pcl = np.load(path)
    elif path[-3:] == 'txt':
        pcl = np.loadtxt(path)
    elif path[-3:] == 'ply':
        pcl = o3d.t.io.read_point_cloud(path)
        pcl = pcl.point.positions.numpy()
    else:
        raise ValueError('input file should be txt, ply or npy')
    
    data = {
        'point': pcl[:, :3],
        'feat': None,
        'label': np.zeros((pcl.shape[0]), dtype=np.int32)
    }
    return data

if __name__ == "__main__":
    torch.set_num_threads(1)

    path = str(here("data/0_raw/tls_tropical_leaf_wood/test/dro_033_pc.txt"))
    print("Reading:", path)
    tile_old = read_cloud_old(path)
    print("Starting inference (num_workers=0)...")
    result = pipeline.run_inference(tile_old)
    print("Inference finished")
    # optional: inspect or save result