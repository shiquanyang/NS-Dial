import json
import numpy as np
from utils.config import *
from utils.utils_temp import entityList, get_type_dict, get_img_fea, load_img_fea
import logging
from utils.utils_general_reasoning_for_synthetic_dataset import *
import ast


def read_langs(file_name, global_entity, type_dict, img_path, max_line=None):
    print("Reading lines from {}".format(file_name))
    data, context_arr, conv_arr, kb_arr, img_arr, facts_arr, conclusion_arr, conclusion_label_arr = [], [], [], [], [], [], [], []
    max_res_len, sample_counter, turn = 0, 0, 0
    image_feas = {}
    with open(file_name) as fin:
        cnt_lin = 1
        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split('\t', 1)
                if int(nid) >= 10:  # for restaurant and hotel domains.
                # if int(nid) >= 12:  # for movie domain.
                    try:
                        line_list = line.split('\t')
                        if len(line_list) == 6:
                            u, r, gold_ent, conclusion, query_type, num_hops = line_list[0], line_list[1], line_list[2], line_list[3], line_list[4], line_list[5]
                        elif len(line_list) == 7:
                            u, r, gold_ent, _, conclusion, query_type, num_hops = line_list[0], line_list[1], line_list[2], line_list[3], line_list[4], line_list[5], line_list[6]
                    except:
                        print(line)
                        continue
                    gen_u = generate_memory(u, "$u", str(turn), image_feas)
                    if len(gen_u[0]) > 4:
                        print(gen_u)
                        print(u, r)
                    context_arr += gen_u
                    conv_arr += gen_u
                    # u_token = u.split(' ')
                    # conv_arr += u_token
                    ptr_index, ent_words = [], []

                    structure_type = []
                    if query_type == '1':
                        structure_type.append(0)
                    elif query_type == '3':
                        structure_type.append(1)

                    target_conclusion = ast.literal_eval(conclusion)

                    query_entity_h = []
                    query_entity_t = []
                    if query_type == '1':
                        query_entity_h.append(target_conclusion[2])
                        query_entity_t.append(target_conclusion[1])
                    elif query_type == '3':
                        query_entity_h.append(target_conclusion[0])
                        query_entity_t.append(target_conclusion[1])

                    all_entities = generate_all_entities(facts_arr)

                    # add positive conclusion and negative conclusion
                    conclusion_arr.append(target_conclusion)
                    conclusion_label_arr.append(1.0)
                    neg_cnt = 0
                    for idx, entity in enumerate(all_entities):
                        if entity != target_conclusion[0] and entity != target_conclusion[2] and neg_cnt < args['max_neg_cnt']:
                            if int(query_type) == 1:
                                neg_triple = [entity, target_conclusion[1], target_conclusion[2]]
                            elif int(query_type) == 3:
                                neg_triple = [target_conclusion[0], target_conclusion[1], entity]
                            conclusion_arr.append(neg_triple)
                            conclusion_label_arr.append(0.0)
                            neg_cnt += 1

                    ent_words = ast.literal_eval(gold_ent)

                    for key in r.split():
                        if key in global_entity and key not in ent_words:
                            ent_words.append(key)
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in global_entity)]
                        index = max(index) if (index) else len(context_arr)
                        ptr_index.append(index)

                    candidates_pointer = [1 if triple[0] in ent_words else 0 for triple in kb_arr]

                    data_detail = {
                        'context_arr':list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]),  # dialogue history + kb
                        'response':r,  # response
                        'ptr_index':ptr_index+[len(context_arr)],
                        'ent_index':ent_words,
                        'conv_arr':list(conv_arr),  # dialogue history ---> bert encode
                        'kb_arr':list(kb_arr),  # kb ---> memory encode
                        'img_arr':list(img_arr),  # image ---> attention encode
                        'conclusion_arr':list(conclusion_arr),
                        'facts_arr':list(facts_arr),
                        'conclusion_label_arr':conclusion_label_arr,
                        'id':int(sample_counter),
                        'ID':int(cnt_lin),
                        'domain':"",
                        'turns':[turn],
                        'structure_type':list(structure_type),
                        'query_entity_h':list(query_entity_h),
                        'query_entity_t':list(query_entity_t),
                        'candidates_pointer':list(candidates_pointer)}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(turn), image_feas)
                    if len(gen_r[0]) > 4:
                        print(gen_r)
                        print(u, r)
                    context_arr += gen_r
                    conv_arr += gen_r
                    # r_token = r.split(' ')
                    # conv_arr += r_token
                    if max_res_len < len(r.split()):
                        max_res_len = len(r.split())
                    sample_counter += 1
                    turn += 1
                    conclusion_arr, conclusion_label_arr = [], []
                else:
                    r = line
                    fact_info = r.split('\t')
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
                context_arr, conv_arr, kb_arr, img_arr, facts_arr = [], [], [], [], []
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


def generate_memory(sent, speaker, time, image_feas):
    sent_new = []
    if speaker == "$u" or speaker == "$s":
        sent_token = sent.split(' ')
    else:
        sent_token = sent.split('\t')
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
    file_train = '/home/shiquan/Projects/DialogueReasoning/data/synthetic/restaurant/restaurant_domain_generated_samples_trn.txt'
    file_dev = '/home/shiquan/Projects/DialogueReasoning/data/synthetic/restaurant/restaurant_domain_generated_samples_dev.txt'
    file_test = '/home/shiquan/Projects/DialogueReasoning/data/synthetic/restaurant/restaurant_domain_generated_samples_tst.txt'
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
