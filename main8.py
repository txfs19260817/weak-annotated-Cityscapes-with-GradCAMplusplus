import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
from glob import glob

from tqdm import tqdm

from cam import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM
from utils import city_palette, prepare_input, transform, label2idx, idx2label, CityscapesClassificationDataset

parser = argparse.ArgumentParser(description='Generate weak annotations. ')
parser.add_argument('--root', type=str, default="G:/Dataset/Cityscapes/leftImg8bit/train", help='dataset path')
parser.add_argument('--output', type=str, default="G:\\Projects\\GRADCAMPP\\output", help='output path')
parser.add_argument('--weight', type=str, default="G:\\Projects\\Grad-CAM-semantic-mask\\model_best_resnest101.pth.tar",
                    help='weight path')


def main(args):
    # const
    threshold_up = [
        .7,  # "road",
        .7,  # "sidewalk",
        .4,  # "building",
        .5,  # "wall",
        .6,  # "fence",
        .65,  # "pole",
        .65,  # "traffic light",
        .65,  # "traffic sign",
        .4,  # "vegetation",
        .7,  # "terrain",
        .4,  # "sky",
    ]

    threshold_down = [
        .4,  # "road",
        .4,  # "sidewalk",
        .7,  # "building",
        .5,  # "wall",
        .6,  # "fence",
        .65,  # "pole",
        .65,  # "traffic light",
        .65,  # "traffic sign",
        .7,  # "vegetation",
        .4,  # "terrain",
        .7,  # "sky",
    ]

    # model
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=False)
    model.fc = nn.Linear(2048, 19, bias=True)
    model.load_state_dict(torch.load(args.weight)['state_dict'])
    model.eval()
    model = model.cuda()

    # CAM
    target_layer = model.layer4[2].conv3
    wrapped_model = GradCAMpp(model, target_layer)

    # dataset
    files = [f for f in glob(args.root + '/**', recursive=True) if os.path.isfile(f)]
    for filename in tqdm(files):
        origin_img = Image.open(filename)
        inputs = prepare_input(np.array(origin_img))
        inputs = inputs.view(3, 1024, 4, 512).permute(2, 0, 1, 3).reshape(4, 3, 2, 512, 512).permute(0, 2, 1, 3, 4).reshape(
            8, 3, 512, 512)

        masks = []
        for i in range(inputs.shape[0]):
            if i % 2 == 0:  # up part
                threshold = threshold_up
            else:
                threshold = threshold_down
            tensor = inputs[i].unsqueeze(0).cuda()
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
                area[idx] = (c >= threshold[indices[idx]]).sum()
            order = area.argsort()[::-1]
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask.fill(255)
            for o in order:
                c, idx = cams[o], indices[o]
                mask[c >= threshold[idx]] = idx
            masks.append(mask)
        out_array = np.hstack([np.vstack((masks[i], masks[i + 1])) for i in range(0, inputs.shape[0], 2)])
        out_img = Image.fromarray(out_array)
        out_img.putpalette(city_palette)
        out_img.save(os.path.join(args.output, filename.split('\\')[-1]))


if __name__ == "__main__":
    main(parser.parse_args())
