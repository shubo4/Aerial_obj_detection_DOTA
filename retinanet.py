import math
import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchvision.models.detection.image_list import ImageList
import random

from torchvision.ops import boxes as box_ops, misc as misc_nn_ops, sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.transforms._presets import ObjectDetection
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _COCO_CATEGORIES
from torchvision.models._utils import _ovewrite_value_param, handle_legacy_interface
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import _box_loss, overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

class Rhead(RetinaNetClassificationHead):
  def __init__(self, in_channels,num_anchors,num_classes,prior_probability=0.01,alpha =0.75):
    super(Rhead,self).__init__(in_channels,num_anchors,num_classes,prior_probability=0.01,alpha =0.75)


    def compute_loss(self, targets, head_outputs, matched_idxs,alpha=alpha):
      losses = []

      cls_logits = head_outputs["cls_logits"]

      for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
        
          # determine only the foreground
          foreground_idxs_per_image = matched_idxs_per_image >= 0
          num_foreground = foreground_idxs_per_image.sum()

          # create the target classification
          gt_classes_target = torch.zeros_like(cls_logits_per_image)
          gt_classes_target[
              foreground_idxs_per_image,
              targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
          ] = 1.0

          # find indices for which anchors should be ignored
          valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

          # compute the classification loss
          losses.append(
              sigmoid_focal_loss(
                  cls_logits_per_image[valid_idxs_per_image],
                  gt_classes_target[valid_idxs_per_image],
                  alpha = 0.75,
                  reduction="sum",
              )
              / max(1, num_foreground)
          )

      return _sum(losses) / len(targets)
        
class RetinaNet_(nn.Module):
    __annotations__ = {
            "box_coder": det_utils.BoxCoder,
            "proposal_matcher": det_utils.Matcher,
        }

    def __init__(
        self,
        backbone,
        num_classes,
        # transform parameters
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.4, #0.5
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
        **kwargs,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        if not isinstance(anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"anchor_generator should be of type AnchorGenerator or None instead of {type(anchor_generator)}"
            )

        if anchor_generator is None:
            anchor_generator = _default_anchorgen()
        self.anchor_generator = anchor_generator

        if head is None:
            head = Rhead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], num_classes,prior_probability=0.01,alpha =0.75)
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        "Expected target boxes to be a tensor of shape [N, 4].",
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        #images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

#         # Converting tuple of images in single tensor representing a batch
        image_batch = model.transform.batch_images(images)

#         # Create an ImageList
        images = ImageList(image_batch, [(image.shape[-2], image.shape[-1]) for image in images])

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                # compute the losses
                losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
            #detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)


_COMMON_META = {
    "categories": _COCO_CATEGORIES,
    "min_size": (1, 1),
}

