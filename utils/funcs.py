import torch
from torchvision.ops.boxes import box_area
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision
from copy import deepcopy


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

# modified from torchvision to also return the union


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def format_nuim_targets(tgts, device):
    new_targets = []

    def extract_labels_and_boxes(tgt):
        labels = []
        boxes = []
        if tgt.shape[0] == 0:
            return None, None
        for i in range(len(tgt)):
            tensor_label_box = tgt[i].to(device)
            tensor_label = tensor_label_box[0].long()
            tensor_box = tensor_label_box[1:].float()

            labels.append(tensor_label)
            boxes.append(tensor_box)

        return torch.stack(labels, dim=0), torch.stack(boxes, dim=0)

    for tgt in tgts:
        labels, boxes = extract_labels_and_boxes(tgt)
        if labels is None:
            continue
        new_targets.append({"labels": labels, "boxes": boxes})
    return new_targets


def generate_trace_report(model, device, batch_size=20, input_size=(800, 800), filename="trace.json"):
    input_data = torch.randn(batch_size, 3, input_size(0), input_size(1)).to(device)
    model = deepcopy(model.to(device))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
                 record_shapes=True) as prof:
        _ = model(input_data)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=5))
    prof.export_chrome_trace(filename)