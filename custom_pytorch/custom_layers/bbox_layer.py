import torch
from torch import nn
from torch.nn import Conv2d, Linear, Sigmoid
import cv2
import numpy as np
import torch.nn.functional as F


class BBoxLayer(nn.Module):
    def __init__(self, in_channels, features_map_dimensions, metadata_dimension, max_dim=32):
        super().__init__()
        self.outputs = None
        assert len(features_map_dimensions) == 3, features_map_dimensions
        # conv 1x1 to change features channels size to 4
        self.mapping_conv = Conv2d(features_map_dimensions[0], 4 * 4, 1)
        self.linear_map_dimensions = (min(max_dim, features_map_dimensions[1]), min(
            max_dim, features_map_dimensions[2]))
        self.mapping_linear = Linear(features_map_dimensions[1] * features_map_dimensions[2],
                                     self.linear_map_dimensions[0] *
                                     self.linear_map_dimensions[1])
        self.mapping_activation = Sigmoid()
        self.metadata_linear = Linear(metadata_dimension, self.linear_map_dimensions[0]
                                      * self.linear_map_dimensions[1])
        self.metadata_activation = Sigmoid()

        self.inputs_conv = nn.Sequential(Conv2d(in_channels, out_channels=4, kernel_size=3),
                                         nn.ZeroPad2d((1, 1, 1, 1)))
        self.inputs_activation = Sigmoid()
        self.bboxes_layer = Conv2d(
            4, out_channels=4, kernel_size=features_map_dimensions[-2:])
        # self.bboxes_activation = Sigmoid()
        pr = features_map_dimensions[-2] - 1
        pc = features_map_dimensions[-1] - 1
        self.padding = nn.ZeroPad2d(
            (pr // 2, pr - pr // 2, pc // 2, pc - pc // 2))

    def forward(self, inputs, feature_map_outputs, metadata_inputs):
        if len(inputs.size()) == 2:
            inputs = inputs.view((1, 1) + inputs.size())
        if len(inputs.size()) == 3:
            inputs = inputs.view((1,) + inputs.size())
        convolved_feats = self.mapping_conv(feature_map_outputs)
        meta_feats = self.metadata_linear(
            metadata_inputs.view(metadata_inputs.size()[:2] + (-1,))).view(
                (metadata_inputs.size()[:2] + self.linear_map_dimensions[:2]))
        linear_feats = convolved_feats @ meta_feats
        convolved_inputs = self.inputs_activation(self.inputs_conv(inputs))
        b, c, h, w = convolved_inputs.size()

        mapped_inputs = self.padding(torch.stack([F.conv2d(
            convolved_input.unsqueeze(0),
            self.mapping_activation(linear_feat).view(
                4, 4, linear_feats.size()[-2], linear_feats.size()[-1]))
            for convolved_input, linear_feat in zip(convolved_inputs, linear_feats)]))

        mapped_inputs = mapped_inputs.view(
            b, c, mapped_inputs.size()[-2], mapped_inputs.size()[-1])
        self.outputs = self.padding(  # self.bboxes_activation( # We dont use it as we have logits bce loss
            self.bboxes_layer(mapped_inputs))  # )  # B * 4 * H * W
        return self.outputs

    def _get_centroids(self, img):
        _, _, _, centroids = cv2.connectedComponentsWithStats(
            (img > 0.5).astype(np.uint8))

        return [tuple(loc) for loc in centroids]

    def _get_bboxes(self, ul_locs, ur_locs, bl_locs, br_locs):

        ul_locs = sorted(ul_locs)
        ur_locs = sorted(ur_locs)
        bl_locs = sorted(bl_locs)
        br_locs = sorted(br_locs)
        ul_cnt = 0
        boxes = []
        bound_ur = []
        bound_bl = []
        bound_br = []
        while ul_cnt < len(ul_locs):
            ul_loc = ul_locs[ul_cnt]
            br_cnt = 0
            while br_cnt < len(br_locs):
                br_loc = br_locs[br_cnt]
                if br_loc > ul_loc and br_loc not in bound_br:
                    bound_br.append(br_loc)
                    break
                br_cnt += 1
            else:
                ul_cnt += 1
                continue
            bl_cnt = 0
            while bl_cnt < len(bl_locs):
                bl_loc = bl_locs[bl_cnt]
                if bl_loc[1] > ul_loc[1] and bl_loc[0] < br_loc[0] and bl_loc not in bound_bl:
                    bound_bl.append(bl_loc)
                    break
                bl_cnt += 1
            else:
                bound_br = bound_br[:-1]
                ul_cnt += 1
                continue
            ur_cnt = 0
            while ur_cnt < len(ur_locs):
                ur_loc = ur_locs[ur_cnt]
                if ur_loc[0] > ul_loc[0] and ur_loc[1] < br_loc[1] and ur_loc not in bound_ur:
                    bound_ur.append(ur_loc)
                    break
                ur_cnt += 1
            else:
                bound_br = bound_br[:-1]
                bound_bl = bound_bl[:-1]
                ul_cnt += 1
                continue
            boxes.append([ul_loc, ur_loc, br_loc, bl_loc])
            ul_cnt += 1
        return boxes

    def get_bboxes(self, outputs=None):
        if outputs is not None:
            self.outputs = outputs
        outputs = self.outputs.to('cpu').data.numpy()
        bboxes = []
        for out in outputs:
            ul_locs = self._get_centroids(out[0, :, :])
            ur_locs = self._get_centroids(out[1, :, :])
            bl_locs = self._get_centroids(out[2, :, :])
            br_locs = self._get_centroids(out[3, :, :])
            bboxes.append(self._get_bboxes(ul_locs, ur_locs, bl_locs, br_locs))
        return bboxes


def create_bboxes_mask_from_mask(mask):
    if len(mask.shape) == 3:
        mask = np.sum(mask.astype(int), axis=2)
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes_mask = np.zeros((4,) + mask.shape, np.uint8)
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        x = min(x, mask.shape[1] - 1 - w)
        y = min(y, mask.shape[0] - 1 - h)

        bboxes_mask[0, y, x] = 1
        bboxes_mask[1, y + h, x] = 1
        bboxes_mask[2, y, x + w] = 1
        bboxes_mask[3, y + h, x + w] = 1
    return bboxes_mask
