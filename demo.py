# coding: UTF-8
import numpy as np
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif, build_detect_set
import torch
from tqdm import tqdm
import jieba

# Command line argument parsing
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="bert", help='choose a model: bert, bert_CNN,'
                                                              'bert_DPCNN,bert_RNN,bert_RCNN,ERNIE')
args = parser.parse_args()

def seg_word(input_file, output_file):
    text_dict = {}
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Load user dictionary and custom words
    jieba.load_userdict("usrlist.txt")
    for word in ['游', '阁', '购', '畅', '拍', '号', '馆']:
        jieba.add_word(word, 100000)

    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
        keyword = line.split('-', 1)[0].split('_', 1)[0].replace(' ', '')
        seg_list = jieba.cut(keyword, cut_all=True)
        result = "Before: " + keyword + " After: " + '/'.join(seg_list)
        text_dict[line] = result

    with open(output_file, 'w', encoding='utf-8') as file:
        for original_line, segmented_result in text_dict.items():
            file.write(f"{original_line} | {segmented_result}\n")


def is_Non_sense(input_file, output_file, original_input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = file.readlines()

    result = []
    original_lines = []
    keywords_to_replace = ["腾讯", "淘宝", "知乎", "阿里", "微信", "百度", "电竞"]

    for index, item in tqdm(enumerate(data), desc="Filtering nonsensical text"):
        after = item.strip().split('After:')[1].strip()

        # Check and replace keywords
        for keyword in keywords_to_replace:
            if keyword in after:
                after = after.replace(keyword, "B")

        parts = after.split('/')
        for i in range(len(parts)):
            if parts[i].isdigit():
                parts[i] = 'D'

        after = '/'.join(parts)

        # Check if there are three consecutive single characters
        if check_consecutive_single_slash(after):
            result.append(after)  # Record the filtered result
            original_lines.append(item.split('|')[0].strip())  # Record only the original content

    with open(output_file, 'w', encoding='utf-8') as file:
        for line in result:
            file.write(f"{line}\n")

    # Output the filtered original content
    with open(original_input_file, 'w', encoding='utf-8') as file:
        for line in original_lines:
            file.write(f"{line}\n")


def check_consecutive_single_slash(input_string):
    parts = input_string.split('/')
    if len(parts) < 3:
        return False
    for i in range(len(parts) - 2):
        if len(parts[i]) == 1 and len(parts[i + 1]) == 1 and len(parts[i + 2]) == 1:
            return True
    return False


if __name__ == '__main__':
    input_file = "datas\data\input.txt"  # Input path
    output_seg_file = "segmented_output.txt"  # Segmentation output file
    output_nonsense_file = "filtered_output.txt"  # Nonsensical text output file
    output_filtered_input_file = "input_.txt"  # Filtered original input file

    # Step 1: Word segmentation
    seg_word(input_file, output_seg_file)

    # Step 2: Nonsensical text filtering
    is_Non_sense(output_seg_file, output_nonsense_file, output_filtered_input_file)

    # Step 3: Model prediction
    dataset = 'datas'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    detect_data = build_detect_set(config)
    print("Length of detect_data:", len(detect_data))
    print("Batch size:", config.batch_size)
    detect_iter = build_iterator(detect_data, config)

    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    predict_all = np.array([], dtype=int)
    print("Get predicted data ready!")
    with torch.no_grad():
        for texts, label in tqdm(detect_iter):
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

    print("Predicted Category:", predict_all)

    # Read segmentation results
    tmp_lst = []
    with open(output_seg_file, "r", encoding='utf-8') as file:
        for item in file:
            tmp_lst.append(item.strip())

    # Output combined results
    with open("res.txt", "w", encoding='utf-8') as file:
        for index, item in enumerate(tmp_lst):
            item = item + '\t' + str(predict_all[index])
            file.write(item + '\n')

    print("Results saved to res_input.txt")