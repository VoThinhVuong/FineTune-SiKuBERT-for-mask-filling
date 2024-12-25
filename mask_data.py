import random
import re
import os

def clean_text(input_text):
    return ''.join([char for char in input_text if is_cjk(char)])

def is_cjk(char):
    # Exclude '冫' (U+51AB) while still checking for other CJK characters
    if char == '冫' or char == "“" or char == "”":
        return False
    return any([
        '\u4E00' <= char <= '\u9FFF',    # CJK Unified Ideographs
        '\u3400' <= char <= '\u4DBF',    # CJK Unified Ideographs Extension A
        '\u20000' <= char <= '\u2A6DF',  # CJK Unified Ideographs Extension B
        '\u2A700' <= char <= '\u2EBEF',  # CJK Unified Ideographs Extension C-F
        '\uF900' <= char <= '\uFAFF',    # CJK Compatibility Ideographs
    ])



def replace_with_mask(chinese_string):
    n = len(chinese_string)

    if n <= 3:
        return []

    def insert_mask(string, indices):
        """Helper function to insert '[MASK]' at specific indices."""
        result = list(string)
        for idx in sorted(indices, reverse=True):
            result[idx] = "[MASK]"
        return "".join(result)

    if 3 < n <= 7:
        indices_set = []
        while len(indices_set) < 2:
            idx = random.randint(0, n - 1)
            if idx not in indices_set:
                indices_set.append(idx)
        replaced_1 = insert_mask(chinese_string, [indices_set[0]]) + f", {chinese_string[indices_set[0]]}"
        replaced_2 = insert_mask(chinese_string, [indices_set[1]]) + f", {chinese_string[indices_set[1]]}"
        return [replaced_1, replaced_2]

    elif 7 < n <= 12:
        indices_set = []
        while len(indices_set) < 2:
            idx = random.randint(0, n - 1)
            if all(abs(idx - other) >= 3 for other in indices_set):
                indices_set.append(idx)
        replaced = insert_mask(chinese_string, indices_set) + f",{chinese_string[indices_set[0]]} {chinese_string[indices_set[1]]}"
        return [replaced]

    elif 12 < n:
        indices_set = []
        while len(indices_set) < 3:
            idx = random.randint(0, n - 1)
            if all(abs(idx - other) >= 3 for other in indices_set):
                indices_set.append(idx)
        replaced = insert_mask(chinese_string, indices_set) + f",{chinese_string[indices_set[0]]} {chinese_string[indices_set[1]]} {chinese_string[indices_set[2]]}"
        return [replaced]


def create_masked_data_file(filename, data_folder = "not_masked_data", dest_folder = "masked_data"):
    new_name = dest_folder + "/" + filename[:-4] + "_MASKED.txt"
    masked_str = []
    with open(data_folder + "/" + filename,"r", encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            #print(line)
            line = clean_text(line)
            masked_str.append(replace_with_mask(line))

    with open(new_name, "w", encoding="utf-8") as file:
        for obj in masked_str:
            for lines in obj:
                file.writelines(lines + "\n")


def mask_all_data(data_folder, dest_folder):
    file_names = os.listdir(data_folder)
    for file in file_names:
        create_masked_data_file(file,data_folder, dest_folder)


def main():
    data_folder = "not_masked_data"
    dest_folder = "masked_data"
    mask_all_data(data_folder,dest_folder)

    file_names = os.listdir(dest_folder)
    with open("data.txt", "w", encoding="utf-8") as file:
        res = ""
        for filename in file_names:
            with open(dest_folder + "/" + filename, "r", encoding="utf-8") as masked_file:
                res += masked_file.read()
        file.write(res)

if __name__ == "__main__":
    main()
