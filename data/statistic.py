import os
from sklearn.model_selection import train_test_split
from utils.tools import get_name


def statistic_data(folders):
    record = ['損傷等級c剥離・鉄筋露出', '範囲外', "don't use", '損傷等級cひびわれ', '損傷等級bひびわれ', '損傷等級e剥離・鉄筋露出', '損傷等級c遊離石灰', '損傷等級e遊離石灰', '損傷等級e 漏水・滞水・土砂詰', '損傷等級dひびわれ', '損傷等級c変形・欠損', '損傷等級d剥離・鉄筋露出', '損傷等級d遊離石灰', '損傷等級d変形・欠損']
    record2 = {}
    for i in range(len(record)):
        record2.update({record[i]: 0})

    for item in folders:
        label_masks = get_name("concrete_data/" + item + "/labels", mode_folder=False)
        for mask in label_masks:
            index = mask.split("_")
            cat_name = index[2][:-4].replace(u'\u3000', '')
            record2[cat_name] += 1

    print(record2)


def main():
    folders_origin = get_name("concrete_data")
    folders_crop = get_name("concrete_cropped")

    train, val = train_test_split(folders_origin, train_size=0.80, random_state=15)
    statistic_data(train)
    statistic_data(val)


if __name__ == '__main__':
    main()
