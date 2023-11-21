import os
import torch
from configs import parser
import torch.nn.functional as F
from model.retrieval.model_main import ModelFusion
from PIL import Image
import numpy as np
from loaders.loader import inference_transformation


@torch.no_grad()
def evaluation(model, img, data, device):
    model.eval()

    data, csv_ = img.to(device, dtype=torch.float32), data.to(device, dtype=torch.float32)
    pred = F.log_softmax(model(data, csv_), dim=-1)
    pred = torch.argmax(pred, dim=-1)[0]
    print("The prediction is category:", pred)


def main():
    # load model and weights
    model = ModelFusion(args)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir, f"{'Fusion' if not args.fusion else 'No_fusion'}_{args.base_model}_cls{args.num_classes}_" + f"cpt{args.num_cpt if not args.pre_train else ''}_" +
                f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)

    img_dir = "data/concrete_cropped_center2/raw/34.73638,135.60429_43_dひびわれ_竪壁/bg_[528, 778].png"

    trans = inference_transformation()
    img = Image.open(img_dir).convert('RGB')
    img = trans(img).unsqueeze(0)

    csv_data = [2017, 58]
    fusion_data = torch.from_numpy(np.array(csv_data)).unsqueeze(0)
    evaluation(model, img, fusion_data, device)


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    args.fusion = True
    main()