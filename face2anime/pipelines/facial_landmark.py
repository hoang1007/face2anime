'''
This script use mmdet and mmpose to detect facial landmark.

Please run following commands to install mmdet and mmpose
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmdet==3.1.0
pip install mmpose==1.1.0
'''

import warnings
from typing import List, Optional, Union
import copy

from PIL import Image
try:
    from mmengine.config import Config
    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.model.utils import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    from mmpose.apis import init_model as init_pose_model
    from mmpose.datasets.datasets.utils import parse_pose_metainfo
    from mmpose.models.builder import build_pose_estimator
    from mmpose.structures import PoseDataSample
    from mmpose.structures.bbox import bbox_xywh2xyxy
except ImportError:
    warnings.warn('Please install mmdet and mmpose at first to run FacialLandmarkPipeline.')

from typing import Optional, Union
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as T


class SimpleFacialLandmarkPipeline:
    def __init__(
        self,
        pose_config_path: str,
        pose_checkpoint_path: str,
        keep_keypoint_ids: Optional[list] = None,
        device: str = 'cpu'
    ):

        self.pose_detector = init_pose_model(pose_config_path, pose_checkpoint_path, device=device)

        if keep_keypoint_ids is not None:
            self.keep_keypoint_ids = np.array(keep_keypoint_ids)
        else:
            self.keep_keypoint_ids = None
    
    def _get_input(self, img: Union[str, np.ndarray, torch.FloatTensor]) -> torch.FloatTensor:
        if isinstance(img, str):
            img = Image.open(img)

        if not isinstance(img, torch.FloatTensor):
            img = T.to_tensor(img)
        return img
    
    def _preprocessing(self, img: torch.FloatTensor):
        device = next(self.pose_detector.parameters()).device
        c, h, w = img.shape
        data_samples = prepare_batch(self.pose_detector, np.zeros((h, w, c)))['data_samples']

        # set bbox scale that is changed by topdown affine transformation
        ds = data_samples[0]
        ds.gt_instances.bbox_scales = np.array([[w, h]]).astype(np.float32)

        if ds.img_shape != ds.input_size:
            inp = nn.functional.interpolate(img.unsqueeze(0), size=ds.input_size).to(device)
        else:
            inp = img.unsqueeze(0).to(device)

        batch = self.pose_detector.data_preprocessor(dict(inputs=inp, data_samples=data_samples))
        return batch

    def predict(self, img: Union[str, np.ndarray, torch.FloatTensor], return_datasample: bool = False):
        img = self._get_input(img)
        batch = self._preprocessing(img)
        pose_res = self.pose_detector(**batch, mode='predict')

        if return_datasample:
            return pose_res
        else:
            result = dict()
            result['keypoints'] = pose_res[0].pred_instances.keypoints[0]
            result['keypoint_scores'] = pose_res[0].pred_instances.keypoint_scores[0]
            try:
                result['heatmaps'] = pose_res[0].pred_fields.heatmaps
            except:
                result['heatmaps'] = None

            if self.keep_keypoint_ids is not None:
                result['keypoints'] = result['keypoints'][self.keep_keypoint_ids]
                result['keypoint_scores'] = result['keypoint_scores'][self.keep_keypoint_ids]
                if result['heatmaps'] is not None:
                    result['heatmaps'] = result['heatmaps'][self.keep_keypoint_ids]

            return result
        
    def __call__(self, img: Union[str, np.ndarray, torch.FloatTensor]):
        img = self._get_input(img)
        batch = self._preprocessing(img)
        heatmaps = self.pose_detector(**batch, mode='tensor')

        if self.keep_keypoint_ids is not None:
            heatmaps = heatmaps[:, self.keep_keypoint_ids]
        
        return heatmaps


def prepare_batch(model: nn.Module,
                    img: Union[np.ndarray, str],
                    bboxes: Optional[Union[List, np.ndarray]] = None,
                    bbox_format: str = 'xyxy') -> List[PoseDataSample]:
    scope = model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    if bboxes is None or len(bboxes) == 0:
        # get bbox from the image size
        if isinstance(img, str):
            w, h = Image.open(img).size
        else:
            h, w = img.shape[:2]

        bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
    else:
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        assert bbox_format in {'xyxy', 'xywh'}, \
            f'Invalid bbox_format "{bbox_format}".'

        if bbox_format == 'xywh':
            bboxes = bbox_xywh2xyxy(bboxes)

    # construct batch data samples
    data_list = []
    for bbox in bboxes:
        if isinstance(img, str):
            data_info = dict(img_path=img)
        else:
            data_info = dict(img=img)
        data_info['bbox'] = bbox[None]  # shape (1, 4)
        data_info['bbox_score'] = np.ones(1, dtype=np.float32)  # shape (1,)
        data_info.update(model.dataset_meta)
        data_list.append(pipeline(data_info))

    return pseudo_collate(data_list)
