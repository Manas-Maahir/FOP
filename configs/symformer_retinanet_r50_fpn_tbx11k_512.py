# SymFormer w/ RetinaNet (full model) for the TB-only PoC (plan.md Phase 7).
# Same recipe as the RetinaNet baseline, but the FPN neck is replaced by SASFPN: FPN followed by a
# shared Symmetric Abnormity Search block (SymAttention + SPE-with-STN, transfer right->left).
#
# This is the PRIMARY comparison: train this and the baseline with the same seed/schedule and
# compare category-agnostic AP50/AP on TB-val.

_base_ = './retinanet_r50_fpn_tbx11k_512.py'

# Replace the plain FPN with SASFPN. The FPN args mirror mmdet's RetinaNet FPN; `sas` configures
# the novel block. Full SymFormer setting = SymAttention + SPE + STN + right->left.
model = dict(
    neck=dict(
        type='SASFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        sas=dict(
            attention='symattention',
            pe='spe',
            use_stn=True,
            direction='r2l',
            num_heads=8,
            num_points=4,
            offset_scale=0.1,
        )))
