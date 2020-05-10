import argparse
import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import os
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM
from utils import city_palette, prepare_input, transform, label2idx, idx2label, CityscapesClassificationDataset

parser = argparse.ArgumentParser(description='Generate weak annotations. ')
parser.add_argument('--root', type=str, default="E:/", help='dataset path')
parser.add_argument('--output', type=str, default="G:\\Projects\\GRADCAMPP\\output", help='output path')
parser.add_argument('--weight', type=str, default="G:\\Projects\\Grad-CAM-semantic-mask\\model_best_resnest101.pth.tar",
                    help='weight path')


def main(args):
    # const
    threshold = [
        .5,  # "road",
        .5,  # "sidewalk",
        .5,  # "building",
        .5,  # "wall",
        .7,  # "fence",
        .7,  # "pole",
        .7,  # "traffic light",
        .7,  # "traffic sign",
        .5,  # "vegetation",
        .5,  # "terrain",
        .5,  # "sky",
    ]

    # dataset
    filename = "sample/aachen_000001_000019_leftImg8bit.png"
    origin_img = Image.open(filename)
    inputs = prepare_input(np.array(origin_img))

    # model
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=False)
    model.fc = nn.Linear(2048, 19, bias=True)
    model.load_state_dict(torch.load(args.weight)['state_dict'])
    model.eval()
    model = model.cuda()

    # CAM
    target_layer = model.layer4[2].conv3
    wrapped_model = GradCAMpp(model, target_layer)

    tensor = inputs.cuda()
    target = model(tensor).cpu().sigmoid()
    indices = (target[0][:11] > 0.5).nonzero().view(-1).tolist()
    cams = []
    for j in indices:
        cam, idx = wrapped_model(tensor, idx=j)
        cam = nn.functional.interpolate(cam.cpu(), size=tuple(tensor.size()[-2:]), mode='bilinear',
                                        align_corners=False)
        cam = cam.squeeze(0).squeeze(0).numpy()
        cams.append(cam)

    area = np.zeros(len(indices), dtype=np.uint32)
    for (idx, c) in enumerate(cams):
        i = indices[idx]
        area[idx] = (c >= threshold[i]).sum()
    order = area.argsort()[::-1]
    mask = np.zeros_like(cams[0], dtype=np.uint8)
    mask.fill(255)
    for o in order:
        c, i = cams[o], indices[o]
        mask[c >= threshold[i]] = i

    out_img = Image.fromarray(mask)
    out_img.putpalette(city_palette)
    out_img.save(os.path.join(args.output, filename.split('/')[-1]))


if __name__ == "__main__":
    main(parser.parse_args())
