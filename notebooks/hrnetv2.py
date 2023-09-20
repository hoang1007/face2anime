default_scope = 'mmpose'

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
        std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
        bgr_to_rgb=False),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            upsample=dict(mode='bilinear', align_corners=False)),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://msra/hrnetv2_w18'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=270,
        out_channels=28,
        deconv_out_channels=None,
        conv_out_channels=(270, ),
        conv_kernel_sizes=(1, ),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        flip_mode='heatmap',
        shift_heatmap=True,
        output_heatmaps=True
    ))

train_dataloader = dict(
    dataset=dict(
        metainfo=dict(from_file='/data1.local/vinhpt/hoangvh/face2anime/notebooks/dataset_info.py'),))

test_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

test_dataloader = dict(
    dataset=dict(
        metainfo=dict(from_file='/data1.local/vinhpt/hoangvh/face2anime/notebooks/dataset_info.py'),
        data_mode='topdown',
        test_mode=True,
        pipeline=test_pipeline,
    ))

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')
