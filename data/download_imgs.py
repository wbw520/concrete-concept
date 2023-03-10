import json
import shutil
import os, base64
import requests
from io import BytesIO
from PIL import Image


def load_json(name):
    with open(name, "r", encoding="utf8") as root_info:
        data = json.load(root_info)
    return data


def get_image(url):
    print(url)
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def main():
    shutil.rmtree('concrete_data/', ignore_errors=True)
    os.makedirs('concrete_data/', exist_ok=True)
    data = load_json(file_name)
    for item in data:
        ID = item["ID"]
        image_url = item["Labeled Data"]
        labelled_objects = item["Label"]["objects"]
        if len(labelled_objects) == 0:
            continue
        name = item["External ID"][:-4]
        print(name)
        os.makedirs('concrete_data/' + name, exist_ok=True)
        image_row = get_image(image_url)
        image_row.save("concrete_data/" + name + "/row_img.png")
        for label in labelled_objects:
            os.makedirs('concrete_data/' + name + "/labels", exist_ok=True)
            feature_id = label["featureId"]
            color = label["color"]
            title = label["title"]
            instance_url = label["instanceURI"]
            image_label = get_image(instance_url)
            image_label.save('concrete_data/' + name + "/labels/" + feature_id + "_" + color + "_" + title + ".png")


if __name__ == '__main__':
    file_name = "export-2022-12-06T05_41_37.401Z.json"
    main()