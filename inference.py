from PIL import Image
import numpy as np
import os
from loaders.loader import get_val_transformations
from model.retrieval.model_main import MainModel
import torch
import shutil
from configs import parser
from math import ceil
import cv2


def patch_calculation(tile_rows, tile_cols, image_source, image_size, model):
    full_probs_pred = torch.from_numpy(np.zeros((args.num_classes, image_size[0], image_size[1]))).to(args.device)
    count_predictions_pred = torch.from_numpy(np.zeros((args.num_classes, image_size[0], image_size[1]))).to(args.device)

    full_probs_cpt = np.zeros((num_cpt, image_size[0], image_size[1]))
    count_predictions_cpt = np.zeros((num_cpt, image_size[0], image_size[1]))

    for row in range(tile_rows):
        print(str(row) + "/" + str(tile_rows))
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            if row == tile_rows - 1:
                y2 = image_size[0]
                y1 = image_size[0] - patch_size
            if col == tile_cols - 1:
                x2 = image_size[1]
                x1 = image_size[1] - patch_size

            cropped_source = Image.fromarray(image_source[y1:y2, x1:x2, :])
            # current_name = str(row) + "_" + str(col)
            # os.makedirs('inference/' + current_name, exist_ok=True)
            # check = 'inference/' + current_name + "/"
            # cropped_source.save(check + "raw.png")

            with torch.set_grad_enabled(False):
                cpt, pred, att, update, normed = model(transform(cropped_source).unsqueeze(0).to(args.device), None, None, check=None)
                pred = pred[0].unsqueeze(-1).unsqueeze(-1).expand(-1, patch_size, patch_size)
                count_predictions_pred[:, y1:y2, x1:x2] += 1
                full_probs_pred[:, y1:y2, x1:x2] += pred

                for id in range(args.num_cpt):
                    slot_image = cv2.resize(normed[id], (224, 224))
                    count_predictions_cpt[id, y1:y2, x1:x2] += 1
                    full_probs_cpt[id, y1:y2, x1:x2] += slot_image

    full_probs_pred /= count_predictions_pred
    _, preds = torch.max(full_probs_pred, 0)

    full_probs_cpt /= count_predictions_cpt
    return preds, full_probs_cpt


def id2trainId(color, label):
    w, h = label.shape
    label_copy = np.zeros((w, h, 3), dtype=np.uint8)
    for index, color in color.items():
        label_copy[label == index] = color

    return label_copy


def color_trans(img):
    color = {0: [0, 0, 0], 1: [244, 35, 232], 2: [220, 20, 60]}
    marked = id2trainId(color, img)
    return marked


def main():
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

    img_dir = "/home/wangbowen/PycharmProjects/concrete/data/concrete_data/2-t-7/row_img.png"
    image_source = np.array(Image.open(img_dir).convert('RGB'))
    height_orl, width_orl, c = image_source.shape
    image_source = cv2.resize(image_source, (int(width_orl / resize_ratio), int(height_orl / resize_ratio)), interpolation=cv2.INTER_LINEAR)
    height, width, c = image_source.shape
    tile_rows = ceil((height - patch_size) / stride) + 1
    tile_cols = ceil((width - patch_size) / stride) + 1
    print("tile_rows:", tile_rows)
    print("tile_cols:", tile_cols)
    preds, full_cpt = patch_calculation(tile_rows, tile_cols, image_source, [height, width], model)

    for i in range(num_cpt):
        save_cpt = full_cpt[i]
        cv2.imwrite(f'inference/{i}_slot.png', save_cpt)

    mask = Image.fromarray(preds.cpu().detach().numpy(), mode='L')
    mask.save(f'inference/prediction.png')


if __name__ == '__main__':
    shutil.rmtree('inference/', ignore_errors=True)
    os.makedirs('inference/', exist_ok=True)
    args = parser.parse_args()
    args.pre_train = False
    save_folder = "concrete_cropped_center/"
    resize_ratio = 2
    patch_size = 224
    stride = 50
    num_cpt = args.num_cpt
    transform = get_val_transformations()
    main()