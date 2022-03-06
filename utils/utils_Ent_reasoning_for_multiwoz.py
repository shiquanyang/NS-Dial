import json
import numpy as np
from utils.config import *
from utils.utils_temp import entityList, get_type_dict, get_img_fea, load_img_fea
import logging
from utils.utils_general_reasoning_for_multiwoz import *
import ast


def read_langs(file_name, global_entity, type_dict, img_path, max_line=None):
    print("Reading lines from {}".format(file_name))
    data, context_arr, conv_arr, conv_arr_plain, kb_arr, img_arr, facts_arr, conclusion_arr, conclusion_label_arr = [], [], [], [], [], [], [], [], []
    max_res_len, sample_counter, turn = 0, 0, 0
    image_feas = {}

    with open('data/multiwoz/multiwoz_entities.json') as f:
        global_entity = json.load(f)

    is_first = True
    with open(file_name) as fin:
        cnt_lin = 1
        for line in fin:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    line = line.replace("#", "")
                    task_type = line
                    continue
                if task_type in ['train', 'restaurant', 'hotel', 'attraction']:
                    nid, line = line.split(' ', 1)
                    if '\t' in line:
                        # deal with dialogue history
                        u, r, gold_ent = line.split('\t')
                        gen_u = generate_memory(u, "$u", str(nid), image_feas)
                        context_arr += gen_u
                        conv_arr += gen_u
                        conv_arr_plain.append(u)

                        # Get gold entity for each domain
                        gold_ent = ast.literal_eval(gold_ent)
                        ent_idx_cal, ent_idx_nav, ent_idx_wet = [], [], []
                        ent_idx_restaurant, ent_idx_hotel, ent_idx_attraction, ent_idx_train, ent_idx_hospital = [], [], [], [], []
                        if task_type == "restaurant":
                            ent_idx_restaurant = gold_ent
                        elif task_type == "hotel":
                            ent_idx_hotel = gold_ent
                        elif task_type == "attraction":
                            ent_idx_attraction = gold_ent
                        elif task_type == "train":
                            ent_idx_train = gold_ent
                        elif task_type == "hospital":
                            ent_idx_hospital = gold_ent
                        ent_index = list(set(ent_idx_restaurant + ent_idx_hotel + ent_idx_attraction + ent_idx_train + ent_idx_hospital))

                        if is_first:
                            kb_arr = kb_arr + [["$$$$"]*MEM_TOKEN_SIZE]
                            is_first = False

                        ptr_index = []
                        for key in r.split():
                            index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                            index = max(index) if (index) else len(context_arr)
                            ptr_index.append(index)

                        sketch_response = generate_template(global_entity, r, gold_ent, kb_arr, task_type)

                        candidates_pointer = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in
                                      context_arr] + [1]

                        data_detail = {
                            'context_arr':list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]),  # dialogue history + kb
                            'response':r,  # response
                            'ptr_index':ptr_index+[len(context_arr)],
                            'ent_index':ent_index,
                            'ent_idx_cal': list(set(ent_idx_cal)),
                            'ent_idx_nav': list(set(ent_idx_nav)),
                            'ent_idx_wet': list(set(ent_idx_wet)),
                            'conv_arr':list(conv_arr),  # dialogue history ---> bert encode
                            'kb_arr':list(kb_arr),  # kb ---> memory encode
                            'img_arr':list(img_arr),  # image ---> attention encode
                            'facts_arr':list(facts_arr),
                            'id':int(sample_counter),
                            'ID':int(cnt_lin),
                            'domain': task_type,
                            'turns':[turn],
                            # 'candidates_pointer':list(candidates_pointer),
                            'candidates_pointer':ptr_index+[len(context_arr)],
                            'ent_idx_restaurant': list(set(ent_idx_restaurant)),
                            'ent_idx_hotel': list(set(ent_idx_hotel)),
                            'ent_idx_attraction': list(set(ent_idx_attraction)),
                            'ent_idx_train': list(set(ent_idx_train)),
                            'ent_idx_hospital': list(set(ent_idx_hospital)),
                            'sketch_response': sketch_response,
                            'global_pointer': list(candidates_pointer),
                            'conv_arr_plain': " ".join(conv_arr_plain)}
                        data.append(data_detail)

                        gen_r = generate_memory(r, "$s", str(turn), image_feas)
                        if len(gen_r[0]) > 4:
                            print(gen_r)
                            print(u, r)
                        context_arr += gen_r
                        conv_arr += gen_r
                        conv_arr_plain.append(r)
                        # r_token = r.split(' ')
                        # conv_arr += r_token
                        if max_res_len < len(r.split()):
                            max_res_len = len(r.split())
                        sample_counter += 1
                        turn += 1
                    else:
                        r = line
                        fact_info = r.split(' ')
                        facts_arr += [fact_info]
                        if "image" not in r:
                            kb_info = generate_memory(r, "", str(nid), image_feas)
                            if len(kb_info[0]) > 4:
                                print(kb_info)
                                print(r)
                            context_arr = kb_info + context_arr
                            kb_arr += kb_info
                        else:
                            image_info = generate_memory(r, "", str(nid), image_feas)
                            img_arr += image_info
            else:
                cnt_lin += 1
                turn = 0
                is_first = True
                context_arr, conv_arr, conv_arr_plain, kb_arr, img_arr, facts_arr = [], [], [], [], [], []
                if(max_line and cnt_lin>max_line):
                    break
    return data, max_res_len


def generate_all_entities(facts_arr):
    entities = []
    for triple in facts_arr:
        subject, object = triple[0], triple[2]
        if subject not in entities:
            entities.append(subject)
        if object not in entities:
            entities.append(object)
    return entities

def generate_template(global_entity, sentence, sent_ent, kb_arr, domain):
    """
    Based on the system response and the provided entity table, the output is the sketch response.
    """
    sketch_response = []
    if sent_ent == []:
        sketch_response = sentence.split()
    else:
        for word in sentence.split():
            if word not in sent_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for key in global_entity.keys():
                    global_entity[key] = [x.lower() for x in global_entity[key]]
                    if word in global_entity[key] or word.replace('_', ' ') in global_entity[key]:
                        ent_type = key
                        break
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response

def generate_memory(sent, speaker, time, image_feas):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:
        try:
            if sent_token[1] == "R_rating":
                sent_token = sent_token + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
            # add logic to cope with image info
            elif sent_token[1].startswith("image"):
                # add image feature retrieve logic
                image_key = sent_token[-1]
                image_fea = get_img_fea(image_key, image_feas)
                sent_token = image_fea
            else:
                sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
            sent_new.append(sent_token)
        except:
            print(sent)
            print(sent_token)
            exit()
    return sent_new


def prepare_data_seq(batch_size=100):
    data_path_babi = '/home/shiquan/Projects/DialogueReasoning/data/dialog-babi'
    img_path = '/Multimodal-Knowledge-Base/images/restaurant'
    file_train = '/home/shiquan/Projects/DialogueReasoning/data/multiwoz/train.txt'
    file_dev = '/home/shiquan/Projects/DialogueReasoning/data/multiwoz/valid.txt'
    file_test = '/home/shiquan/Projects/DialogueReasoning/data/multiwoz/test.txt'
    kb_path = data_path_babi + '-kb-all.txt'
    type_dict = get_type_dict(kb_path, dstc2=False)
    global_ent = entityList(kb_path, 4)

    pair_train, train_max_len = read_langs(file_train, global_ent, type_dict, img_path)
    pair_dev, dev_max_len = read_langs(file_dev, global_ent, type_dict, img_path)
    pair_test, test_max_len = read_langs(file_test, global_ent, type_dict, img_path)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len)

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, lang, max_resp_len


if __name__ == "__main__":
    prepare_data_seq(4)