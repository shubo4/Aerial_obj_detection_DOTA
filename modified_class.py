import torch
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.models.detection import retinanet_resnet50_fpn


class Rhead(RetinaNetClassificationHead):
  def __init__(self, in_channels,
        num_anchors,
        num_classes,
        prior_probability=0.01)
  
class MyRetinaNet(RetinaNet):
  def __init__(self, num_classes, pretrained_backbone=True):
    # Call the constructor of the base class
    super(MyRetinaNet, self).__init__(num_classes=num_classes, pretrained_backbone=pretrained_backbone)

        # You can add your own modifications here if needed

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


# Now you can use my_retinanet for training or inference
