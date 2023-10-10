import os
from sklearn.model_selection import train_test_split
from utils.tools import get_name


def statistic_data(folders):
    record = ['剥離・鉄筋露出', '剥離・鉄筋露出(剥離のみ)', 'ひびわれ', '遊離石灰(つらら状)', '遊離石灰', "範囲外"]
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


def statistic_data2(folders):
    record = ['剥離・鉄筋露出', '剥離・鉄筋露出(剥離のみ)', 'ひびわれ', '遊離石灰(つらら状)', '遊離石灰', "範囲外", "bg"]
    record2 = {}
    for i in range(len(record)):
        record2.update({record[i]: 0})

    for item in folders:
        label_masks = get_name("concrete_cropped_center/raw/" + item, mode_folder=False)
        for mask in label_masks:
            index = mask.split("_")
            cat_name = index[0]
            record2[cat_name] += 1

    print(record2)


def main():
    folders_origin = get_name("concrete_data")
    folders_crop = get_name("concrete_cropped_center/raw")

    statistics_ = folders_crop

    train, val = train_test_split(statistics_, train_size=0.80, random_state=15)

    # statistic_data(train)
    # statistic_data(val)

    statistic_data2(train)
    statistic_data2(val)


if __name__ == '__main__':
    main()
