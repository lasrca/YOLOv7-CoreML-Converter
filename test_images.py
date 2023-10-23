import os
from PIL import Image
import cv2
import coremltools as ct
import numpy as np
import torch
import torchvision
import time
import pandas as pd

labels ={
    0 :'abc' ,
     1:'tnt' ,
     2: 'fox_news',
    3: 'cbs',
    4:'espn',
    5:'msnbc',
    6:'cnn',
     7:"tv"
 }

def load_model(model_path):
    model = ct.models.MLModel(model_path)
    return model


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def run_model_single_image(model, image_filename):
    im0 = cv2.imread(image_filename)
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

    img = letterbox(im0, (640,640), stride=32, auto=False)[0]
    print("img shape:", img.shape)
    print(img[0,0,0])
    # Convert
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    # img = img[:, :, ::-1]
    cv2.imwrite("letter_box_pic.jpg", img)
    img = np.ascontiguousarray(img)
    print(img.shape)
    # im = im[...,::-1]
    # im = cv2.resize(im, (640, 640))
    b = 1
    h, w, ch = img.shape
    im = Image.fromarray((img).astype('uint8'))

    r, g, b = im.getpixel((0, 0))
    # print("RGB")
    # print(r, g, b)
    y = model.predict({'image': im})  # coordinates are xywh normalized
    print(y.keys())
    # print(y["raw_object_confidence"].shape)
    # print(y["raw_label_confidence"].shape)
    # print(y["raw_coordinates"].shape)
    # print(y["raw_confidence"][0,:])
    # print(y["raw_coordinates"][0,:])
    # print(type(y["raw_object_confidence"]))
    # tmp = np.concatenate((y["raw_coordinates"], y["raw_object_confidence"], y["raw_label_confidence"]), axis=1)
    # print(tmp.shape)
    if 'confidence' in y:
    # if "raw_label_confidence" in y :
        box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
        # box = xywh2xyxy(y['raw_coordinates'] * [[w, h, w, h]])  # xyxy pixels
        conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
        y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
        # y = np.concatenate((box,y["raw_object_confidence"], y["raw_label_confidence"]), axis=1)
    else:
         k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
         y = y[k]  # output

    print(y.shape)
    return y


def draw_bbox_image(image_filename, output_model, output_path):
    im0 = cv2.imread(os.path.join("/Users/carlalasry/repos/yolov7/logos_annotations/dataset_tv_and_logos_13062023/val/images/", image_filename))
    im = cv2.resize(im0, (640, 640))
    # print(im.shape)
    detections = []
    if len(output_model)>=1:
        for output in output_model:
            # print(im.shape[:2])
            # print(im0.shape)
            # print(output[ :4])
            output[:4] = scale_coords(im.shape[:2], output[:4], im0.shape).round()
            # print("OUTPUT:", output)
            im0 = cv2.rectangle(im0, (int(output[0]), int(output[1])), (int(output[2]), int(output[3])), (255, 0, 0),  2)
            cv2.putText(im0, labels[int(output[-1])], (int(output[0]), int(output[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            detections.append(labels[int(output[-1])])
        cv2.imwrite(os.path.join(output_path, image_filename), im0)
    else:
        print("Results are None")
        cv2.imwrite(os.path.join(output_path, image_filename), im0)

    return detections


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, classes_conf=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9],
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # print(prediction.shape)
    # print("prediction:")
    # print(prediction)
    nc = prediction.shape[2] - 5  # number of classes
    # print(nc)
    # print(prediction[0,0,...].shape)
    # print(prediction[0,0, ...])
    xc = prediction[..., 4] > conf_thres  # candidates
    # print("candidates:")
    candidates = prediction[xc]
    # print(candidates)
    cnn_candidates = prediction[prediction[..., 10] > conf_thres][:,4]
    # print("CNN CANDIDATES:", cnn_candidates)

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # print(x)

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[:, 4:5] # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
                                 # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # box = xywh2xyxy(x[:, :4])
        box = x[:, :4]
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # print("classes_conf: ", classes_conf)
            # print(x.shape)
            conf, j = x[:, 5:].max(1, keepdim=True)

            if classes_conf is not None:
                conf_thres_list = classes_conf.split(",")  # split conf values by comma
                # print(conf_thres_list)
                conf_thres_list = [float(thres) for thres in conf_thres_list]  # convert conf values to floats
                # print(conf_thres_list)
                # Create the masks according to the conditions
                mask_cls = torch.zeros_like(conf[..., 0], dtype=torch.bool)
                for cls_index, cls_thresh in enumerate(conf_thres_list):
                    mask_cls_cls = (j[..., 0] == cls_index) & (conf[..., 0] > cls_thresh)
                    mask_cls = mask_cls | mask_cls_cls



                mask = conf[..., 0] > conf_thres  # original conf threshold mask
                # mask = mask.unsqueeze(1)
                final_mask = mask & mask_cls  # combine the original and class-specific masks

                # print(mask.shape)
                # print(final_mask.shape)
                x = torch.cat((box, conf, j.float()), 1)[final_mask]
                # x = torch.cat((box, conf, j.float(), mask), 1)[final_mask.flatten()]
                # x = torch.cat((box, conf, j.float(), mask), 1)[final_mask.unsqueeze(0)]

            else:
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # print("STRIDE:", stride)
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    # print("shape:", shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # print("new_shape:", new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # print(r)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # print(new_unpad)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # print(dw, dh)
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        # print("auto:", dw, dh)
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # print("dw, dh:", dw, dh)
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    # print(dw, dh)

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print("Image shape:", img.shape)
    return img, ratio, (dw, dh)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[[0, 2]] -= pad[0]  # x padding
    coords[[1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0] = np.clip(boxes[0], 0, img_shape[1])
    boxes[1] = np.clip(boxes[1], 0, img_shape[0])
    boxes[2] = np.clip(boxes[2], 0, img_shape[1])
    boxes[3] = np.clip(boxes[3], 0, img_shape[0])
    # boxes[0].clamp_(0, img_shape[1])  # x1
    # boxes[ 1].clamp_(0, img_shape[0])  # y1
    # boxes[ 2].clamp_(0, img_shape[1])  # x2
    # boxes[3].clamp_(0, img_shape[0])  # y2


def main():
    # model = load_model("models/output//yolov7_includes_nms_thresholds_0.1.mlmodel")
    model = load_model("models/output/yolov7-iOS_8.mlmodel")
    # model = load_model("models/output/mlmodel_with_output_metadata_label_and_object_conf.mlmodel")
    spec = model.get_spec()
    print(spec.description)
    filenames = []
    results = []
    for filename in os.listdir("/Users/carlalasry/repos/yolov7/logos_annotations/dataset_tv_and_logos_13062023/val/images/"):
        # if filename != "1659195840851423200_frame0.jpg":
        #     continue
        # else:
            print(filename)
            output = run_model_single_image(model, os.path.join("/Users/carlalasry/repos/yolov7/logos_annotations/dataset_tv_and_logos_13062023/val/images/", filename))
            # print(output)
            #
            # output = output.reshape(1,output.shape[0], output.shape[1])
            # print(output.shape)
            # #
            # predictions = torch.from_numpy(output)
            #
            # pred = non_max_suppression(predictions, 0.1, 0.1,
            #                            agnostic=False, classes_conf="0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9")
            #
            # print(pred)

            detections = draw_bbox_image(filename, output, "test_images_mlmodel_with_nms_included_0.1_0.1_quant_fp8/")
            filenames.append(filename)
            results.append(detections)
    df_results = pd.DataFrame.from_dict({"filename": filenames, "results": results})
    df_results.to_csv("results_mlmodel_includes_nms_quant_fp8.csv")

if __name__ == '__main__':
    main()