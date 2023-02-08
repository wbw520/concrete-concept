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
from utils.tools import apply_colormap_on_image
from utils.make_json import make_json_file


def patch_calculation(tile_rows, tile_cols, image_source, image_size, model):
    full_probs_pred = torch.from_numpy(np.zeros((args.num_classes, image_size[0], image_size[1]))).to(args.device)
    count_predictions_pred = torch.from_numpy(np.zeros((args.num_classes, image_size[0], image_size[1]))).to(args.device)

    full_probs_cpt = np.zeros((args.num_cpt, image_size[0], image_size[1]))
    count_predictions_cpt = np.zeros((args.num_cpt, image_size[0], image_size[1]))
    cpt_record = [[] for i in range(args.num_cpt)]
    score_record = [[] for i in range(args.num_cpt)]

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
                pp = torch.argmax(pred, dim=-1)[0]
                pred = pred[0].unsqueeze(-1).unsqueeze(-1).expand(-1, patch_size, patch_size)
                count_predictions_pred[:, y1:y2, x1:x2] += 1
                full_probs_pred[:, y1:y2, x1:x2] += pred
                w = model.state_dict()["cls.weight"][pp]
                w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)
                ccc = np.around(cpt.cpu().detach().numpy(), 4)
                cpt = ccc / 2 + 0.5
                score = (ccc / 2 + 0.5) * w_numpy

                for id in range(args.num_cpt):
                    slot_image = cv2.resize(normed[id], (224, 224))
                    count_predictions_cpt[id, y1:y2, x1:x2] += 1
                    full_probs_cpt[id, y1:y2, x1:x2] += slot_image
                    cpt_record[id].append(cpt[0][id])
                    score_record[id].append(score[0][id])

    final_cpt = []
    final_score = []

    for ii in range(args.num_cpt):
        final_cpt.append((np.mean(cpt_record[ii]) + np.median(cpt_record[ii])) / 2)
        final_score.append((np.mean(score_record[ii]) + np.median(score_record[ii])) / 2)

    full_probs_pred /= count_predictions_pred
    _, preds = torch.max(full_probs_pred, 0)

    full_probs_cpt /= count_predictions_cpt
    return preds, full_probs_cpt, final_cpt, final_score


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

    img_dir = args.inference
    image_source_ = Image.open(img_dir).convert('RGB')
    image_source = np.array(image_source_)
    image_source_.save(args.inference_result_dir + '/origin.png')
    height, width, c = image_source.shape
    tile_rows = ceil((height - patch_size) / stride) + 1
    tile_cols = ceil((width - patch_size) / stride) + 1
    print("tile_rows:", tile_rows)
    print("tile_cols:", tile_cols)
    preds, full_cpt, final_cpt, final_score = patch_calculation(tile_rows, tile_cols, image_source, [height, width], model)

    for i in range(args.num_cpt):
        save_cpt = full_cpt[i]
        cv2.imwrite(args.inference_result_dir + f'/0_slot_{i}.png', save_cpt)

    mask = Image.fromarray(preds.cpu().detach().numpy(), mode='L')
    mask.save(args.inference_result_dir + f'/prediction.png')
    make_json_file(args, final_cpt, final_score)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(args.inference_result_dir + f'/0_slot_{id}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(image_source_, slot_image, 'jet')
        heatmap_on_image.save(args.inference_result_dir + f'/0_slot_mask_{id}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    shutil.rmtree(args.inference_result_dir, ignore_errors=True)
    os.makedirs(args.inference_result_dir, exist_ok=True)
    args.pre_train = False
    save_folder = "concrete_cropped_center/"
    patch_size = 224
    stride = 50
    transform = get_val_transformations()
    main()