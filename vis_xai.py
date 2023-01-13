from model.retrieval.model_main import MainModel
from configs import parser
import torch
import os
from PIL import Image
import numpy as np
from torchcam.cams import CAM, GradCAM
from utils.tools import apply_colormap_on_image, make_grad
from loaders.loader import load_all_imgs, get_val_transformations
import torch.nn.functional as F
import shutil


def main():
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    print("All category:")
    print(cat)
    transform = get_val_transformations()

    # load model and weights
    model = MainModel(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
                                         f"{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                                         f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    index = args.index

    # data = imgs_database[index]
    # label = labels_database[index]
    data = imgs_val[index]
    label = labels_val[index]
    # print("-------------------------")
    # print("label true is: ", cat[label])
    print("-------------------------")
    print(data)
    name = "2-t-7/2_[1117, 2842].png"
    data = "data/concrete_cropped_center/raw/" + name
    data_mask = "data/concrete_cropped_center/mask/" + name
    img_orl = Image.open(data).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    img_mask = Image.open(data_mask).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    img_orl.save(f'visualization/origin.png')
    img_mask.save(f'visualization/origin_mask.png')

    # Visualization Setting
    if vis_style == "attributive":
        RESNET_CONFIG = dict(input_layer='conv1', conv_layer='back_bone', fc_layer='fc')
        MODEL_CONFIG = {**RESNET_CONFIG}
        conv_layer = MODEL_CONFIG['conv_layer']
        input_layer = MODEL_CONFIG['input_layer']
        fc_layer = MODEL_CONFIG['fc_layer']
        cam_extractors = {"CAM": CAM(model, conv_layer, fc_layer), "GradCAM": GradCAM(model, conv_layer)}
        vis_mode = "GradCAM"

        # Calculation
        output1 = model(transform(img_orl).unsqueeze(0).to(device))
        print(output1.shape)
        output = F.softmax(output1, dim=1)
        print("The Model Out of Softmax:", output)
        pred = torch.argmax(output).item()
        print("Predicted as: ", pred + 1)
        make_grad(args, pred, cam_extractors[vis_mode], output1, img_orl, grad_min_level, vis_mode)


if __name__ == '__main__':
    shutil.rmtree('visualization/', ignore_errors=True)
    os.makedirs('visualization/', exist_ok=True)
    grad_min_level = 0
    vis_style = "attributive"
    args = parser.parse_args()
    args.pre_train = True
    main()