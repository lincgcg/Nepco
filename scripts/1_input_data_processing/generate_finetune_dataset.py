#!/usr/bin/python3
#-*- coding:utf-8 -*-
import json
import os
import sys
import csv
import random
import argparse

import numpy as np
import tqdm
from scapy.utils import PcapReader
from sklearn.model_selection import StratifiedShuffleSplit

csv.field_size_limit(sys.maxsize)

dataset_dir = ''

def write_dataset_tsv(data, label, file_dir, split):
    dataset_file = [["label", "text_a"]]
    for index in range(len(label)):
        dataset_file.append([label[index], data[index]])
    os.makedirs(file_dir, exist_ok=True)
    with open(os.path.join(file_dir, split + "_dataset.tsv"), 'w', newline='', encoding="utf-8") as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows(dataset_file)
    return 0

def hex_to_tokens(hex_string, token_nibbles):
    usable_len = len(hex_string) - (len(hex_string) % token_nibbles)
    return [hex_string[i:i + token_nibbles] for i in range(0, usable_len, token_nibbles)]


def get_feature_packet(label_pcap, packet_hex_chars, token_nibbles, max_packets_per_flow):
    
    if os.path.getsize(label_pcap) == 0:
        print("Current File Size = 0 !")
        return -1
    
    feature_data = []
    flow_data_string = ''

    with PcapReader(label_pcap) as pcap_reader:
        for num_packet, packet in enumerate(pcap_reader):
            if max_packets_per_flow > 0 and num_packet >= max_packets_per_flow:
                break

            if 'Ethernet' in packet:
                packet['Ethernet'].src = "00:00:00:00:00:00"
                packet['Ethernet'].dst = "00:00:00:00:00:00"
            if 'IP' in packet:
                packet['IP'].src = "0.0.0.0"
                packet['IP'].dst = "0.0.0.0"
            if 'TCP' in packet:
                packet['TCP'].sport = 0
                packet['TCP'].dport = 0

            packet_hex = bytes(packet).hex()[:packet_hex_chars]
            packet_tokens = hex_to_tokens(packet_hex, token_nibbles)
            if packet_tokens:
                flow_data_string += " ".join(packet_tokens)
                flow_data_string += " "


    flow_data_string = flow_data_string.rstrip()
    feature_data.append(flow_data_string)
    
    if len(flow_data_string) == 0:
        return -1

    return feature_data

def read_data_from_json(json_data, samples):
    X,Y = [], []
    ablation_flag = 1
    features = ["payload"]
    for feature_index in range(len(features)):
        x = []
        label_count = 0
        for label in json_data.keys():
            sample_num = json_data[label]["samples"]
            if X == []:
                if not ablation_flag:
                    y = [label] * sample_num
                    Y.append(y)
                else:
                    if sample_num > 5000:
                        y = [label] * 5000
                    else:
                        y = [label] * sample_num
                    Y.append(y)
            if samples[label_count] < sample_num:
                x_label = []
                for sample_index in random.sample(list(json_data[label][features[feature_index]].keys()),5000):
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            else:
                x_label = []
                for sample_index in json_data[label][features[feature_index]].keys():
                    x_label.append(json_data[label][features[feature_index]][sample_index])
                x.append(x_label)
            label_count += 1
        X.append(x)
    return X,Y

def obtain_data(pcap_path, samples, dataset_save_path, json_data = None):
    
    if json_data:
        X,Y = read_data_from_json(json_data,samples)
    else:
        print("read dataset from json file.")
        with open(dataset_save_path + "dataset.json","r") as f:
            dataset = json.load(f)
        X,Y = read_data_from_json(dataset,samples)

    for index in range(len(X)):
        if len(X[index]) != len(Y):
            print("data and labels are not properly associated.")
            print("x:%s\ty:%s"%(len(X[index]),len(Y)))
            return -1
    return X,Y

def generation(random_seed, pcap_path, samples, dataset_save_path,
               packet_hex_chars=256, token_nibbles=4, max_packets_per_flow=5):

    os.makedirs(dataset_save_path, exist_ok=True)

    if os.path.exists(dataset_save_path + "dataset.json"):
        print("the pcap file of %s is finished generating."%pcap_path)
        X, Y = obtain_data(pcap_path, samples, dataset_save_path)
        return X,Y

    dataset = {}
    label_name_list = []
    session_pcap_path  = {}

    for parent, dirs, files in os.walk(pcap_path):
        if label_name_list == []:
            label_name_list.extend(dirs)
        label_name_list.sort()
        for dir in label_name_list:
            for p,dd,ff in os.walk(parent + "/" + dir):
                session_pcap_path[dir] = pcap_path + dir
        break

    label_id = {}
    for index in range(len(label_name_list)):
        label_id[label_name_list[index]] = index

    r_file_record = []
    print("\nBegin to generate features.")

    label_count = 0
    print(session_pcap_path.keys())
    for key in tqdm.tqdm(session_pcap_path.keys()):

        if label_id[key] not in dataset:
            dataset[label_id[key]] = {
                "samples": 0,
                "payload": {}
            }

        target_all_files = [x[0] + "/" + y for x in [(p, f) for p, d, f in os.walk(session_pcap_path[key])] for y in x[1]]
        random.seed(random_seed)
        target_all_files.sort()
        
        r_files = random.sample(target_all_files, samples[label_count] if len(target_all_files) > samples[label_count] else len(target_all_files) )

        label_count += 1
        for r_f in tqdm.tqdm(r_files):

            feature_datas = get_feature_packet(r_f, packet_hex_chars, token_nibbles, max_packets_per_flow)

            if feature_datas == -1:
                continue
            
            r_file_record.append(r_f)
            for feature_data in feature_datas:
                dataset[label_id[key]]["samples"] += 1
                if len(dataset[label_id[key]]["payload"].keys()) > 0:
                    dataset[label_id[key]]["payload"][str(dataset[label_id[key]]["samples"])] = feature_data
                else:
                    dataset[label_id[key]]["payload"]["1"] = feature_data

    all_data_number = 0
    for index in range(len(label_name_list)):
        print("%s\t%s\t%d"%(label_id[label_name_list[index]], label_name_list[index], dataset[label_id[label_name_list[index]]]["samples"]))
        all_data_number += dataset[label_id[label_name_list[index]]]["samples"]
    print("all\t%d"%(all_data_number))

    with open(dataset_save_path + "picked_file_record","w+") as p_f:
        for i in r_file_record:
            p_f.write(i)
            p_f.write("\n")
    with open(dataset_save_path + "dataset.json", "w+") as f:
        json.dump(dataset,fp=f,ensure_ascii=False,indent=4)

    X,Y = obtain_data(pcap_path, samples, dataset_save_path, json_data = dataset)
    
    
    return X,Y

def unlabel_data(label_data):
    nolabel_data = ""
    with open(label_data,newline='') as f:
        data = csv.reader(f,delimiter='\t')
        for row in data:
            nolabel_data += row[1] + '\n'
    nolabel_file = label_data.replace("test_dataset","nolabel_test_dataset")
    with open(nolabel_file, 'w',newline='') as f:
        f.write(nolabel_data)
    return 0

def models_deal(X_dataset, Y_dataset, x_payload_train, x_payload_test, x_payload_valid, y_train, y_test, y_valid):
    save_dir = dataset_dir
    write_dataset_tsv(x_payload_train, y_train, save_dir, "train")
    write_dataset_tsv(x_payload_test, y_test, save_dir, "test")
    write_dataset_tsv(x_payload_valid, y_valid, save_dir, "valid")
    print("finish generating fine-tuning datagram dataset.\nPlease check in %s" % save_dir)
    unlabel_data(dataset_dir + "test_dataset.tsv")

def dataset_extract(random_seed, pcap_path, samples, dataset_save_path, class_num,
                    packet_hex_chars, token_nibbles, max_packets_per_flow):
    
    print("enter dataset_extract") 
    
    print("calling dataset_generation.generation")
    X,Y = generation(random_seed, pcap_path, samples, dataset_save_path,
                     packet_hex_chars, token_nibbles, max_packets_per_flow)
    print("finish dataset_generation.generation")
    
    dataset_statistic = [0] * class_num
    X_payload= []
    Y_all = []
    
    for app_label in Y:
        for label in app_label:
            Y_all.append(int(label))
    for label_id in range(class_num):
        for label in Y_all:
            if label == label_id:
                dataset_statistic[label_id] += 1
    print("category flow")
    for index in range(len(dataset_statistic)):
        print("%s\t%d" % (index, dataset_statistic[index]))
    print("all\t%d" % (sum(dataset_statistic)))
    

    for index_label in range(len(X[0])):
        for index_sample in range(len(X[0][index_label])):
            X_payload.append(X[0][index_label][index_sample])

    # train : valid : test = 8 : 1 : 1
    split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=41) 
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 

    x_payload = np.ones((len(X_payload), 1))
    dataset_label = np.asarray(Y_all)

    x_payload_train = []
    y_train = []

    x_payload_valid = []
    y_valid = []

    x_payload_test = []
    y_test = []
    
    for train_index, test_index in split_1.split(x_payload, dataset_label):
        x_payload_train, y_train = [X_payload[i] for i in train_index], dataset_label[train_index]
        x_payload_test, y_test =  [X_payload[i] for i in test_index], dataset_label[test_index]
        
    for test_index, valid_index in split_2.split(x_payload_test, y_test):
        x_payload_valid, y_valid = [x_payload_test[i] for i in valid_index], y_test[valid_index]
        x_payload_test, y_test = [x_payload_test[i] for i in test_index], y_test[test_index]


    del x_payload

    if not os.path.exists(dataset_save_path+"dataset/"):
        os.makedirs(dataset_save_path+"dataset/", exist_ok=True)

    output_x_payload_train = os.path.join(dataset_save_path + "dataset/", 'x_datagram_train.npy')
    output_x_payload_test = os.path.join(dataset_save_path + "dataset/", 'x_datagram_test.npy')
    output_x_payload_valid = os.path.join(dataset_save_path + "dataset/", 'x_datagram_valid.npy')
    output_y_train = os.path.join(dataset_save_path+"dataset/",'y_train.npy')
    output_y_test = os.path.join(dataset_save_path + "dataset/", 'y_test.npy')
    output_y_valid = os.path.join(dataset_save_path + "dataset/", 'y_valid.npy')


    x_payload_train_memmap = np.memmap(output_x_payload_train,dtype='U2564',mode= "w+",shape=(len(x_payload_train),))
    x_payload_train_memmap[:] = x_payload_train

    x_payload_test_memmap = np.memmap(output_x_payload_test,dtype='U2564',mode= "w+",shape=(len(x_payload_test),))
    x_payload_test_memmap[:] = x_payload_test

    x_payload_valid_memmap = np.memmap(output_x_payload_valid,dtype='U2564',mode= "w+",shape=(len(x_payload_valid),))
    x_payload_valid_memmap[:] = x_payload_valid

    y_train_memmap = np.memmap(output_y_train,dtype='int32',mode= "w+",shape=(len(y_train),))
    y_train_memmap[:] = y_train

    y_test_memmap = np.memmap(output_y_test, dtype='int32',mode= "w+",shape=(len(y_test),))
    y_test_memmap[:] = y_test

    y_valid_memmap = np.memmap(output_y_valid,dtype='int32',mode= "w+",shape=(len(y_valid),))
    y_valid_memmap[:] = y_valid


    X_dataset = {}
    Y_dataset = {}

    
    models_deal(X_dataset, Y_dataset, x_payload_train, x_payload_test, x_payload_valid, y_train, y_test, y_valid)

def main():
    
    parser = argparse.ArgumentParser(description='Test for argparse')
    
    parser.add_argument("--dataset_dir", type=str,
                        help='''Path of the tsv file for fine-tune")''')
    parser.add_argument("--middle_save_path", type=str,
                        help='''Path of the middle files")''')
    parser.add_argument("--class_num", type=int,
                        help='''class number(e.g., 20)''')
    parser.add_argument("--random_seed", type=int,
                        help='''random seed''')
    parser.add_argument("--pcap_path", type=str,
                        help='''Path of the pcap dataset path''')
    parser.add_argument("--packet_hex_chars", type=int, default=256,
                        help='''Max hexadecimal characters kept per packet''')
    parser.add_argument("--token_nibbles", type=int, default=4,
                        help='''Hex characters per token. Use 4 for hex_vocab.txt''')
    parser.add_argument("--max_packets_per_flow", type=int, default=5,
                        help='''Max packets read from each pcap; <=0 means all''')
    
    args = parser.parse_args()
    
    samples = [5000]
    samples = samples * args.class_num
    
    global dataset_dir
    dataset_dir = args.dataset_dir
    
    if args.packet_hex_chars <= 0:
        raise ValueError("--packet_hex_chars must be positive.")
    if args.token_nibbles <= 0:
        raise ValueError("--token_nibbles must be positive.")

    dataset_extract(args.random_seed, args.pcap_path, samples, args.middle_save_path, args.class_num,
                    args.packet_hex_chars, args.token_nibbles, args.max_packets_per_flow)

if __name__ == '__main__':
    main()
