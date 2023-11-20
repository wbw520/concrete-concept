from PIL import Image
import numpy as np
import os
import shutil
from math import ceil
import cv2
from utils.tools import get_name


def patch_calculation(tile_rows, tile_cols, datas, image_source, save_root):

    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            center_x = (x2 + x1) // 2
            center_y = (y2 + y1) // 2

            for i in range(len(datas)):
                mask = datas[i][0]
                label_copy = datas[i][1]
                mask_cat = datas[i][2]

                cropped_center = mask[center_y - center_size//2: center_y + center_size//2, center_x - center_size//2: center_x + center_size//2, :]
                crop_mask = mask[y1:y2, x1:x2, :]
                center_count = np.sum(np.logical_and(*list([cropped_center[:, :, i] == color[i] for i in range(3)])))

                if np.random.randint(0, 3) == 1 and center_count == center_size**2:
                    cropped_source = image_source[y1:y2, x1:x2, :]
                    cropped_copy = label_copy[y1:y2, x1:x2, :]
                    save_img_mask = cv2.addWeighted(cropped_source, 0.8, cropped_copy, 0.5, 0)
                    save_img = Image.fromarray(cropped_source)
                    save_img_mask = Image.fromarray(save_img_mask)
                    save_img.save(save_folder + "raw/" + save_root + "/" + mask_cat + "_" + str([center_y, center_x]) + ".png")
                    save_img_mask.save(save_folder + "mask/" + save_root + "/" + mask_cat + "_" + str([center_y, center_x]) + ".png")
                    break

                if np.random.randint(0, 10) == 1 and np.sum(crop_mask) == 0:
                    cropped_source = image_source[y1:y2, x1:x2, :]
                    save_img = Image.fromarray(cropped_source)
                    save_img.save(save_folder + "raw/" + save_root + "/" + "bg_" + str([center_y, center_x]) + ".png")
                    save_img.save(save_folder + "mask/" + save_root + "/" + "bg_" + str([center_y, center_x]) + ".png")
                    break


def extract_crop(folders):
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder + 'mask', exist_ok=True)
    os.makedirs(save_folder + 'raw', exist_ok=True)
    cat_list = []
    color_list = []
    print(len(folders))
    for item in folders:
        print(item)
        os.makedirs(save_folder + "mask/" + item, exist_ok=True)
        os.makedirs(save_folder + "raw/" + item, exist_ok=True)
        image_source = np.array(Image.open(root + item + "/row_img.png").convert('RGB'))
        h, w, c = image_source.shape
        tile_rows = ceil((h - patch_size) / stride)
        tile_cols = ceil((w - patch_size) / stride)
        mask_list = []
        mask_img_name = get_name(root + item + "/labels", mode_folder=False)

        for mask in mask_img_name:
            names = mask.split("_")
            label_name = names[0]
            color_name = names[1].lstrip('#')
            color_name = tuple(int(color_name[i:i+2], 16) for i in (0, 2, 4))
            cat_name = names[2][:-4].replace(u'\u3000', '')

            if cat_name not in cat_list:
                cat_list.append(cat_name)
                color_list.append(color_name)

            image_label = np.array(Image.open(root + item + "/labels/" + mask).convert('RGB'))
            label_copy = np.zeros((h, w, 3), dtype=np.uint8)
            label_copy[np.logical_and(*list([image_label[:, :, i] == color[i] for i in range(3)]))] = color_name
            mask_list.append([image_label, label_copy, cat_name])

        patch_calculation(tile_rows, tile_cols, mask_list, image_source, item)

    print(cat_list)
    print(color_list)


def main():
    folders = get_name(root[:-1])
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder, exist_ok=True)
    extract_crop(folders)


if __name__ == '__main__':
    root = "concrete_data2/"
    save_folder = "concrete_cropped_center2/"
    patch_size = 256
    center_size = 10
    stride = 50
    color = [255, 255, 255]
    main()