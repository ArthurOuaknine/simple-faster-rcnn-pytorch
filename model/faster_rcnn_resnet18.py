import torch as t
from torch import nn
from torchvision.models import resnet18
from model.custom_resnet import resnet18 as custom_resnet18
from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import opt

def set_bn_fix(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad = False

def decom_resnet18():

    if opt.caffe_pretrain:
        model = resnet18(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = custom_resnet18(pretrained=False)

    # freeze top conv and bn layers
    # for p in model.conv1.parameters(): p.requires_grad = False
    # for p in model.layer1.parameters(): p.requires_grad = False
    # model.apply(set_bn_fix)

    # resnet.layer0 to resnet.layer3 for extractor
    features_extractor = nn.Sequential(model.conv1, model.bn1, model.leaky_relu,
                                       model.maxpool, model.layer1,
                                       model.layer2, model.layer3)

    # layer4 for classifier
    features_classifier = nn.Sequential(model.layer4)

    return features_extractor, features_classifier

class FasterRCNNRESNET18(FasterRCNN):
    """Faster R-CNN based on RESNET18.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 8  # downsample 8x with resnet18

    def __init__(self,
                 n_fg_class=20,
                 # ratios=[0.5, 1, 2],
                 # anchor_scales=[8, 16, 32]
                 ratios=[0.125, 0.25, 0.5, 1, 2],
                 anchor_scales=[0.125, 0.25, 0.5, 2, 4, 8, 16]
                 # ratios=[0.01, 0.015, 0.02, 0.025, 0.03, 0.03125, 0.035, 0.125, 0.25, 0.5, 1, 2],
                 # anchor_scales=[0.125, 0.25, 0.5, 2, 4, 8, 16]
                 ):
                 
        extractor, classifier = decom_resnet18()

        rpn = RegionProposalNetwork(
            256, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = RESNET18RoIHead(
            n_class=n_fg_class + 1,
            roi_size=14,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        # FLAG: testing values (ON ALL DATASETS !!! KEEP TRAINING)
        # to put in the super()
        loc_normalize_mean = (86.58434286, 27.9424, 95.64971429, 35.6096)
        loc_normalize_std = (56.39664352, 9.54124262, 54.85805827, 9.01109407)

        super(FasterRCNNRESNET18, self).__init__(
            extractor,
            rpn,
            head
            # loc_normalize_mean,
            # loc_normalize_std
        )


class RESNET18RoIHead(nn.Module):
    """Faster R-CNN Head for RESNET18 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): layer4 feature Linear ported from resnet101

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(RESNET18RoIHead, self).__init__()

        self.classifier = classifier
        # self.cls_loc = nn.Linear(2048, n_class * 4)
        # self.score = nn.Linear(2048, n_class)
        self.cls_loc = nn.Linear(512, n_class * 4)
        self.score = nn.Linear(512, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        feature = self.classifier(pool)
        feature = self.avgpool(feature)

        feature = feature.view(feature.size(0), -1)
        roi_cls_locs = self.cls_loc(feature)
        roi_scores = self.score(feature)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
