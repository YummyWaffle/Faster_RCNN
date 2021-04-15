# Dataset Params.
class_num = 2 # Contains Background
# Backbone Params.
network = 'vgg16'
feat_stride = 32
# RPN Params.
rpn_inplane = 512
base_size = 16
ratios = [0.5,1.,2.]
#scales = [1.,2.,4.]
scales = [2.,4.,8.]
pre_nms_topk = 300
aft_nms_topk = 200
roi_pool_size = 7
spatial_scale = 1./float(feat_stride)
rpn_nms_thr = 0.7
# Cls-Reg Net Params
in_feats = roi_pool_size**2 * rpn_inplane
# Running Params.
