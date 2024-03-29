import random
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations import functional as F


def union_of_bboxes(bboxes, erosion_rate=0.0):
    """Calculate union of bounding boxes.

    Args:
        bboxes (list): List like bounding boxes. Format is `[x_min, y_min, x_max, y_max]`.
        erosion_rate (float): How much each bounding box can be shrinked,
            useful for erosive cropping.
            Set this in range [0, 1]. 0 will not be erosive at all, 1.0
            can make any bbox to lose its volume.
    """
    x1, y1 = 1, 1
    x2, y2 = 0, 0
    for b in bboxes:
        w, h = (b[2] - b[0]), (b[3] - b[1])
        lim_x1, lim_y1 = b[0] + erosion_rate * w, b[1] + erosion_rate * h
        lim_x2, lim_y2 = b[2] - erosion_rate * w, b[3] - erosion_rate * h
        x1, y1 = np.min([x1, lim_x1]), np.min([y1, lim_y1])
        x2, y2 = np.max([x2, lim_x2]), np.max([y2, lim_y2])
    return x1, y1, x2, y2


class RandomBBoxSafeCrop(DualTransform):
    """Crop a random part of the input without loss of bboxes, if they exist, otherwise fallbacks to
    random cropping. If the requested random part size is larger than the union
    of the bounding boxes,
    a part of them will be cropped.

    Args:
        height (int): height after crop.
        width (int): width after crop.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the
        interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
            cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, erosion_rate=0.0, interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.erosion_rate = erosion_rate

    def apply(self, img, x_min, y_min, x_max, y_max, interpolation=cv2.INTER_LINEAR, **params):
        crop = F.clamping_crop(img, x_min, y_min, x_max, y_max)
        return F.resize(crop, self.height, self.width, interpolation)

    def squeeze_or_expand(self, x, x2, expected_ratio, current_ratio, length):
        if expected_ratio > current_ratio:
            x_expand = expected_ratio - current_ratio
            x2_expanded = min(x2 + x_expand * random.random(), 1)
            x_expanded = max(0, x - (x_expand - (x2_expanded - x2)))
            w_start = x_expanded * length
            width = length * (x2_expanded - x_expanded)
        else:
            x_squeeze = current_ratio - expected_ratio
            assert x2 > x
            quant = x_squeeze * random.random()
            x2_squeezed = x2 - quant
            assert x2_squeezed > 0
            x_squeezed = x + x_squeeze - quant
            assert x2_squeezed > x_squeezed, (x2_squeezed, x_squeezed)
            w_start = x_squeezed * length
            width = length * (x2_squeezed - x_squeezed)
        return w_start, w_start + width

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params['image'].shape[:2]
        assert 'bboxes' in params
        expected_h_ratio = self.height / img_h
        expected_w_ratio = self.width / img_w
        if params['bboxes']:
            # get union of all bboxes
            x, y, x2, y2 = union_of_bboxes(bboxes=params['bboxes'],
                                           erosion_rate=self.erosion_rate)
            current_w_ratio = x2 - x
            current_h_ratio = y2 - y
            x_min, x_max = self.squeeze_or_expand(
                x, x2, expected_w_ratio, current_w_ratio, img_w)
            y_min, y_max = self.squeeze_or_expand(
                y, y2, expected_h_ratio, current_h_ratio, img_h)
            ret = {'x_min': x_min, 'y_min': y_min,
                   'x_max': x_max, 'y_max': y_max}
        else:
            width = self.width
            height = self.height
            x_min = (1 - expected_w_ratio) * random.random() * img_w
            y_min = (1 - expected_h_ratio) * random.random() * img_h
            ret = {'x_min': x_min, 'y_min': y_min,
                   'x_max': x_min + width,
                   'y_max': y_min + height}
        return ret

    def apply_to_bbox(self, bbox, x_min, y_min, x_max, y_max, rows=0, cols=0, **params):
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, rows, cols)

    def apply_to_mask(self, mask, x_min, y_min, x_max, y_max, rows=0, cols=0,
                      interpolation=cv2.INTER_NEAREST, **params):
        crop = F.clamping_crop(mask, x_min, y_min, x_max, y_max)
        return F.resize(crop, self.height, self.width, interpolation)

    @property
    def targets_as_params(self):
        return ['image', 'bboxes', 'mask']

    def get_transform_init_args_names(self):
        return ('height', 'width', 'erosion_rate', 'interpolation')
