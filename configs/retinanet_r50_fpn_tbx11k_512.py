# RetinaNet baseline for the TB-only PoC (plan.md Phase 5).
# ResNet-50 + FPN, 512x512, SGD, batch 8, 24 epochs, category-agnostic (1-class) TB detection.
#
# This inherits mmdetection's stock RetinaNet and overrides the dataset, input size, schedule,
# and class count. It targets **mmdetection 3.x**. Override `data_root` at launch, e.g.:
#   python tools/train.py configs/retinanet_r50_fpn_tbx11k_512.py \
#       --cfg-options data_root=/content/drive/MyDrive/tbx11k_tb512/
#
# The category-agnostic annotations (single "TB" class) are used so COCO AP/AP50 is the primary
# metric directly. Active/latent per-type eval would instead use the 2-class JSONs (num_classes=2).

_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

# make the SASFPN neck importable (harmless for the baseline, needed by the SymFormer configs)
custom_imports = dict(imports=['symformer_tb.mmdet_plugin'], allow_failed_imports=False)

data_root = 'data/tbx11k_tb512/'          # OVERRIDE at launch (Drive path)
classes = ('TB',)
image_size = (512, 512)

# ---- model: single-class detector -----------------------------------------------------
model = dict(bbox_head=dict(num_classes=1))

# ---- data pipeline --------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),          # paper's only augmentation: random horizontal flip
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=8, num_workers=2,               # drop to 4/2 if the T4 OOMs (and scale LR)
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/tb_train_agnostic.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        type='CocoDataset', data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/tb_val_agnostic.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/tb_val_agnostic.json',
    metric='bbox', classwise=False)
test_evaluator = val_evaluator

# ---- schedule (paper stage-1 detection: SGD, 24 epochs) --------------------------------
# base lr 0.01 is tuned for effective batch 16; linear-scale to batch 8 -> 0.005.
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001))
train_cfg = dict(max_epochs=24, val_interval=1)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[16, 22], gamma=0.1),
]

# ---- checkpoint / resume / reproducibility (plan.md: survive T4 time-outs) -------------
# max_keep_ckpts=1: a checkpoint is ~300MB (model + optimizer). Resuming only needs the latest,
# and evaluation only needs the final one, so keeping 1 bounds Drive usage (free Drive is 15GB;
# 14 runs x 3 kept would be ~12GB).
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1,
                                     save_last=True))
# launch training with `--resume` so the latest checkpoint on Drive is picked up automatically.
randomness = dict(seed=0, deterministic=False)
load_from = None
