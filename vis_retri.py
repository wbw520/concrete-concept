from model.retrieval.model_main import MainModel
from configs import parser
import torch
import os
from PIL import Image
import numpy as np
from utils.tools import apply_colormap_on_image
from loaders.loader import load_all_imgs, get_val_transformations
import h5py
import shutil


shutil.rmtree('vis/', ignore_errors=True)
shutil.rmtree('vis_pp/', ignore_errors=True)
os.makedirs('vis/', exist_ok=True)
os.makedirs('vis_pp/', exist_ok=True)
np.set_printoptions(suppress=True)


def main():
    # load all imgs
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    print("All category:")
    print(cat)
    transform = get_val_transformations()

    # load model and weights
    model = MainModel(args, vis=True)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    index = args.index

    data = imgs_val[index]
    label = labels_val[index]
    print("-------------------------")
    print("label true is: ", cat[label])
    print("-------------------------")

    img_orl = Image.open(data).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    # img_mask = Image.open(data_mask).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
    img_orl.save(f'vis/origin.png')
    # img_mask.save(f'vis/origin_mask.png')
    cpt, pred, att, update, normed = model(transform(img_orl).unsqueeze(0).to(device), None, None)
    print("-------------------------")
    pp = torch.argmax(pred, dim=-1)
    print(pred)
    print("predicted as: ", cat[pp])

    w = model.state_dict()["cls.weight"][label]
    w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)
    ccc = np.around(cpt.cpu().detach().numpy(), 4)
    # draw_bar(w_numpy, name)

    print("--------weight---------")
    print(w_numpy)

    print("--------cpt---------")
    print(ccc/2 + 0.5)

    print("------importance--------")
    print((ccc/2 + 0.5) * w_numpy)

    if args.use_weight:
        w[w < 0.1] = 0
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), w)

    for id in range(args.num_cpt):
        slot_image = np.array(Image.open(f'vis/0_slot_{id}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
        heatmap_on_image.save("vis/" + f'0_slot_mask_{id}.png')

    # get retrieval cases
    f1 = h5py.File(f"data_map/{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", 'r')
    database_hash = f1["database_hash"]
    database_labels = f1["database_labels"]
    test_hash = f1["test_hash"]
    test_labels = f1["test_labels"]

    print("-------------------------")
    print("generating concept samples")

    for j in range(args.num_cpt):
        root = 'vis_pp/' + "cpt" + str(j) + "/"
        os.makedirs(root, exist_ok=True)
        selected = np.array(database_hash)[:, j]
        ids = np.argsort(-selected, axis=0)
        idx = ids[:args.top_samples]
        for i in range(len(idx)):
            current_is = idx[i]
            category = cat[int(database_labels[current_is][0])]
            img_orl = Image.open(imgs_database[current_is]).convert('RGB')
            imgs_unit1 = imgs_database[current_is].split("/")[-1]
            imgs_unit2 = imgs_database[current_is].split("/")[-2]
            img_mask_url = "data/concrete_cropped_center/mask/" + imgs_unit2 + "/" + imgs_unit1
            img_mask = Image.open(img_mask_url).convert('RGB')
            img_mask = img_mask.resize([224, 224], resample=Image.BILINEAR)

            img_orl = img_orl.resize([224, 224], resample=Image.BILINEAR)
            cpt, pred, att, update, normed = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category, j])
            img_orl.save(root + f'/orl_{i}_{category}.png')
            img_mask.save(root + f'/made_mask_{i}_{category}.png')
            slot_image = np.array(Image.open(root + f'mask_{i}_{category}.png'), dtype=np.uint8)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
            heatmap_on_image.save(root + f'jet_{i}_{category}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    args.pre_train = False
    args.index = 500
    main()