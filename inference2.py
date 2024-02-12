import os
import torch
from configs import parser
import torch.nn.functional as F
from model.retrieval.model_main import ModelFusion
from PIL import Image
import numpy as np
from utils.tools import apply_colormap_on_image
import shutil
from loaders.loader import inference_transformation
import matplotlib.pyplot as plt


shutil.rmtree('vis/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)


def draw_bar(data, name):
    plt.figure(figsize=(10, 6), dpi=80)
    font = 22
    x_bar = np.arange(0, len(data), 1)
    plt.bar(x_bar, data)
    plt.ylabel('Weight', fontsize=font)
    plt.xlabel('Concepts', fontsize=font)
    plt.xticks(list(x_bar))
    plt.tick_params(labelsize=22)
    plt.tight_layout()
    plt.savefig(name + "weight.pdf", bbox_inches="tight")
    plt.show()


@torch.no_grad()
def evaluation(model, img, data, device):
    model.eval()

    data, csv_ = img.to(device, dtype=torch.float32), data.to(device, dtype=torch.float32)
    cls, cat = model(data, csv_)

    pred = F.log_softmax(cls, dim=-1)
    pred = torch.argmax(pred, dim=-1)[0]
    print("The prediction is category:", pred)

    w = model.state_dict()["cls.weight"][pred]
    w_numpy = np.around(torch.tanh(torch.relu(w)).cpu().detach().numpy(), 4)
    ccc = np.around(cat.cpu().detach().numpy(), 4)

    print(w_numpy)
    print(ccc)
    draw_bar(w_numpy, "importance")


def main():
    # load model and weights
    model = ModelFusion(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir, f"{'Fusion' if not args.fusion else 'No_fusion'}_{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)

    img_dir = "data/concrete_cropped_center2/raw/34.73638,135.60429_43_dひびわれ_竪壁/bg_[528, 778].png"

    trans = inference_transformation()
    img_orl = Image.open(img_dir).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    img_orl.save(f'vis/origin.png')
    img = trans(img_orl).unsqueeze(0)


    csv_data = [2017, 58]
    fusion_data = torch.from_numpy(np.array(csv_data)).unsqueeze(0)
    evaluation(model, img, fusion_data, device)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
        heatmap_on_image.save("vis/" + f'0_slot_mask_{id}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    args.fusion = True
    main()