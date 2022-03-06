import torch
import torch.nn as nn
from models.ReasonEngine import ReasonEngine
from models.MultiTaskQuestionGeneratorForDecoder import MultiTaskQuestionGeneratorForDecoder
from models.ReasonDecoder import ReasonDecoder
from models.Encoder import Encoder
from torch.optim import lr_scheduler
from torch import optim
from utils.config import *
from utils.masked_cross_entropy import *
import numpy as np
import random
from utils.measures import wer, moses_multi_bleu
import json
from utils.utils_general import _cuda


class LTHR(nn.Module):
    def __init__(self, lang, emb_size, hidden_size, max_depth, lr, path, dropout, relations_cnt, entities_cnt):
        super(LTHR, self).__init__()
        self.name = "LTHR"

        self.lang = lang
        self.embed_size = emb_size
        self.hidden_size = hidden_size
        self.max_depth = max_depth
        self.relations_cnt = relations_cnt
        self.entities_cnt = entities_cnt
        self.lr = lr
        self.input_size = lang.n_words
        self.dropout = dropout

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/encoder.th')
                self.question_generator = torch.load(str(path)+'/question_generator.th')
                self.reasoner = torch.load(str(path)+'/reasoner.th')
                self.decoder = torch.load(str(path)+'/decoder.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/encoder.th',lambda storage, loc: storage)
                self.question_generator = torch.load(str(path)+'/question_generator.th',lambda storage, loc: storage)
                self.reasoner = torch.load(str(path)+'/reasoner.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/decoder.th',lambda storage, loc: storage)
        else:
            self.encoder = Encoder(lang.n_words,
                                   hidden_size,
                                   dropout,
                                   lang.n_words,
                                   hidden_size,
                                   lang)
            self.question_generator = MultiTaskQuestionGeneratorForDecoder(lang.n_words,
                                                                           hidden_size,
                                                                           dropout,
                                                                           lang.n_words,
                                                                           hidden_size,
                                                                           lang,
                                                                           args['maxhops'])
            self.reasoner = ReasonEngine(max_depth, emb_size, hidden_size, relations_cnt, entities_cnt, lang)
            self.decoder = ReasonDecoder(self.encoder.embedding, lang, hidden_size, dropout)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.question_generator_optimizer = optim.Adam(self.question_generator.parameters(), lr=lr)
        self.reasoner_optimizer = optim.Adam(self.reasoner.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.reasoner_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.decoder_scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion = nn.NLLLoss()
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.question_generator.cuda()
            self.reasoner.cuda()
            self.decoder.cuda()

    def reset(self,):
        self.loss, self.loss_re, self.loss_v, self.loss_st, self.loss_qe_h, self.loss_qe_t, self.loss_ca, self.reinforce_loss, self.loss_g, self.loss_mm, self.print_every = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1

    def print_loss(self):
        print_loss_avg = self.loss / self.print_every
        print_loss_re = self.loss_re / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_ca = self.loss_ca / self.print_every
        print_loss_g = self.loss_g / self.print_every
        self.print_every += 1
        return 'L:{:.2f}, LV:{:.2f}, LG:{:.2f}, LCA:{:.2f}, LRE:{:.2f}'.format(print_loss_avg, print_loss_v, print_loss_g, print_loss_ca, print_loss_re)

    def save_model(self, dec_type):
        if args['dataset'] == 'reasoning':
            name_data = "DialogueReasoning/"
        elif args['dataset'] == 'multiwoz':
            name_data = "DialogueReasoningMultiWOZ/"
        elif args['dataset'] == 'kvr':
            name_data = "DialogueReasoningSMD/"
        directory = 'save/LTHR-' + args["addName"] + name_data + 'HDD' + str(
            self.hidden_size) + 'BSZ' + str(args['batch']) + 'lr' + str(
            self.lr) + str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/encoder.th')
        torch.save(self.question_generator, directory + '/question_generator.th')
        torch.save(self.reasoner, directory + '/reasoner.th')
        torch.save(self.decoder, directory + '/decoder.th')

    def unk_mask(self, data):
        story_size = data['context_arr'].size()
        rand_mask = np.ones(story_size)
        bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
        rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
        conv_rand_mask = np.ones(data['conv_arr'].size())
        for bi in range(story_size[0]):
            start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
            conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
        if USE_CUDA:
            rand_mask = torch.Tensor(rand_mask).cuda()
            conv_rand_mask = torch.Tensor(conv_rand_mask).cuda()
        else:
            rand_mask = torch.Tensor(rand_mask)
            conv_rand_mask = torch.Tensor(conv_rand_mask)
        conv_story = data['conv_arr'] * conv_rand_mask.long()
        story = data['context_arr'] * rand_mask.long()
        return story, conv_story

    def split_samples(self, triples_score, triples_label):
        batch_size, max_len = triples_score.shape[0], triples_score.shape[1]
        all_neg_labels, all_neg_scores, pos_neg_labels, pos_neg_scores = [], [], [], []
        for bt in range(batch_size):
            for i in range(max_len):
                if 1.0 in triples_label[bt, i]:
                    pos_neg_labels.append(triples_label[bt, i])
                    pos_neg_scores.append(triples_score[bt, i])
                else:
                    all_neg_labels.append(triples_label[bt, i])
                    all_neg_scores.append(triples_score[bt, i])
        all_neg_scores = torch.stack(all_neg_scores, dim=0)
        all_neg_labels = torch.stack(all_neg_labels, dim=0)
        if len(pos_neg_scores) != 0:
            pos_neg_scores = torch.stack(pos_neg_scores, dim=0)
            pos_neg_labels = torch.stack(pos_neg_labels, dim=0)
        else:
            pos_neg_scores, pos_neg_labels = None, None
        return all_neg_labels, all_neg_scores, pos_neg_labels, pos_neg_scores

    def train(self, data, clip, reset=0):
        if reset:
            self.reset()

        # Zero gradients of optimizers
        self.encoder_optimizer.zero_grad()
        self.question_generator_optimizer.zero_grad()
        self.reasoner_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        batch_size = len(data['context_arr_lengths'])

        if args['unk_mask']:
            context_arr, conv_arr = self.unk_mask(data)
        else:
            context_arr, conv_arr = data['context_arr'], data['conv_arr']

        # Encode dialogue history and generate questions for reasoning engine
        hidden = self.encoder(conv_arr, data['conv_arr_lengths'])
        encoded_hidden_t = hidden

        candidates_prob_global = self.question_generator.compute_candidates_probability(context_arr, data['context_arr_lengths'], encoded_hidden_t, None, False)

        # Decoding
        max_target_length = max(data['response_lengths'])
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        conv_arr_plain, response_plain,\
            kb_arr_plain = data['conv_arr_plain'], data['response_plain'], data['facts_arr_plain']
        outputs_vocab, decoded_fine, decoded_coarse, candidates_prob, triples_score, triples_label = self.decoder(
            encoded_hidden_t,
            self.question_generator,
            self.reasoner,
            data['sketch_response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            False,
            context_arr,
            data['context_arr_lengths'],
            data['facts_arr'],
            data['ent_index'],
            candidates_prob_global,
            conv_arr_plain,
            response_plain,
            kb_arr_plain)

        # Calculate loss and backpropagation
        # Reasoner Loss
        triples_label, triples_score = triples_label.transpose(0, 1), triples_score.transpose(0, 1)
        mask = torch.zeros_like(triples_score)
        for bt in range(batch_size):
            end = data['response_lengths'][bt]
            mask[bt, :end, :] = torch.ones([end, triples_score.shape[2]])
        masked_triples_score = triples_score * mask
        masked_triples_label = triples_label * mask
        masked_triples_score = masked_triples_score.reshape(-1, triples_score.shape[2])
        masked_triples_label = masked_triples_label.reshape(-1, triples_label.shape[2])
        loss_re = self.criterion_bce(masked_triples_score, masked_triples_label)
        loss_v = masked_cross_entropy(
            outputs_vocab.transpose(0, 1).contiguous(),
            data['sketch_response'].contiguous(),
            data['response_lengths'])
        loss_ca = masked_cross_entropy(
            candidates_prob.transpose(0, 1).contiguous(),
            data['candidates_pointer'].contiguous(),
            data['response_lengths']
        )
        loss_g = self.criterion_bce(candidates_prob_global, data['global_pointer'].float())

        loss = loss_v + loss_ca + loss_g + loss_re

        loss.backward()

        # Clip gradient norms
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.question_generator.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.reasoner.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.question_generator_optimizer.step()
        self.reasoner_optimizer.step()
        self.decoder_optimizer.step()

        self.loss_v += loss_v.item()
        self.loss_ca += loss_ca.item()
        self.loss_g += loss_g.item()
        self.loss_re += loss_re.item()
        self.loss += loss.item()

    def evaluate(self, data, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.question_generator.train(False)
        self.reasoner.train(False)
        self.decoder.train(False)

        pbar = tqdm(enumerate(data), total=len(data))

        ref, hyp = [], []
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred, F1_pred_reasoning, F1_pred_babi, F1_restaurant_pred, F1_hotel_pred, F1_attraction_pred, F1_train_pred = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count, F1_count_reasoning, F1_count_babi, F1_restaurant_count, F1_hotel_count, F1_attraction_count, F1_train_count = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        global_entity_list = []
        dialog_acc_dict = {}
        nb_samples, total, acc = 0, 0, 0

        if args['dataset'] == 'multiwoz':
            with open('data/multiwoz/multiwoz_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                global_entity_list = list(set(global_entity_list))
        elif args['dataset'] == 'kvr':
            with open('data/smd/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        for j, batch_data in pbar:
            batch_size = len(batch_data['context_arr_lengths'])
            # Encode dialogue history and generate questions for reasoning engine
            hidden = self.encoder(batch_data['conv_arr'], batch_data['conv_arr_lengths'])
            encoded_hidden_t = hidden

            candidates_prob_global = self.question_generator.compute_candidates_probability(
                batch_data['context_arr'], batch_data['context_arr_lengths'], encoded_hidden_t, None, False)

            # Decoding
            max_target_length = max(batch_data['response_lengths'])
            use_teacher_forcing = False
            conv_arr_plain, response_plain,\
                kb_arr_plain = batch_data['conv_arr_plain'], batch_data['response_plain'], batch_data['facts_arr_plain']
            outputs_vocab, decoded_fine, decoded_coarse, candidates_prob, triples_score, triples_label = self.decoder(
                encoded_hidden_t,
                self.question_generator,
                self.reasoner,
                batch_data['sketch_response'],
                max_target_length,
                batch_size,
                use_teacher_forcing,
                True,
                batch_data['context_arr'],
                batch_data['context_arr_lengths'],
                batch_data['facts_arr'],
                batch_data['ent_index'],
                candidates_prob_global,
                conv_arr_plain,
                response_plain,
                kb_arr_plain)

            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = batch_data['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)

                if args['dataset'] == 'kvr':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(batch_data['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_cal'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_nav'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_wet'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                elif args['dataset'] == 'multiwoz':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(batch_data['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])  # data[14]: ent_index, data[9]: kb_arr_plain.
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_restaurant'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])  # data[28]: ent_idx_restaurant, data[9]: kb_arr_plain.
                    F1_restaurant_pred += single_f1
                    F1_restaurant_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_hotel'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])  # data[29]: ent_idx_hotel, data[9]: kb_arr_plain.
                    F1_hotel_pred += single_f1
                    F1_hotel_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_attraction'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])  # data[30]: ent_idx_attraction, data[9]: kb_arr_plain.
                    F1_attraction_pred += single_f1
                    F1_attraction_count += count
                    single_f1, count = self.compute_prf(batch_data['ent_idx_train'][bi], pred_sent.split(),
                                                        global_entity_list, batch_data['kb_arr_plain'][bi])  # data[31]: ent_idx_train, data[9]: kb_arr_plain.
                    F1_train_pred += single_f1
                    F1_train_count += count
                elif args['dataset'] == 'reasoning':
                    single_f1_reasoning, count_reasoning = self.compute_prf(batch_data['ent_index'][bi], pred_sent.split(), global_entity_list, batch_data['kb_arr_plain'][bi])
                    F1_pred_reasoning += single_f1_reasoning
                    F1_count_reasoning += count_reasoning
                elif args['dataset'] == 'babi':
                    single_f1_babi, count_babi = self.compute_prf(batch_data['ent_index'][bi], pred_sent.split(),
                                                                  global_entity_list, batch_data['kb_arr_plain'][bi])
                    F1_pred_babi += single_f1_babi
                    F1_count_babi += count_babi
                else:
                    # compute Dialogue Accuracy Score
                    current_id = batch_data['ID'][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)

                # compute Per-response Accuracy Score
                total += 1
                if gold_sent == pred_sent:
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, batch_data, pred_sent, pred_sent_coarse, gold_sent)

        self.encoder.train(True)
        self.question_generator.train(True)
        self.reasoner.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)

        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
            print("CAL F1:\t{:.4f}".format(F1_cal_pred / float(F1_cal_count)))
            print("WET F1:\t{:.4f}".format(F1_wet_pred / float(F1_wet_count)))
            print("NAV F1:\t{:.4f}".format(F1_nav_pred / float(F1_nav_count)))
            print("BLEU SCORE:\t" + str(bleu_score))
        elif args['dataset'] == 'multiwoz':
            F1_score = F1_pred / float(F1_count)
            rest_f1 = 0.0 if F1_restaurant_count == 0 else (F1_restaurant_pred / float(F1_restaurant_count))
            hotel_f1 = 0.0 if F1_hotel_count == 0 else (F1_hotel_pred / float(F1_hotel_count))
            attraction_f1 = 0.0 if F1_attraction_count == 0 else (F1_attraction_pred / float(F1_attraction_count))
            train_f1 = 0.0 if F1_train_count == 0 else (F1_train_pred / float(F1_train_count))
            print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
            print("Restaurant F1:\t{:.4f}".format(rest_f1))
            print("Hotel F1:\t{:.4f}".format(hotel_f1))
            print("Attraction F1:\t{:.4f}".format(attraction_f1))
            print("Train F1:\t{:.4f}".format(train_f1))
            print("BLEU SCORE:\t" + str(bleu_score))
        elif args['dataset'] == 'reasoning':
            F1_score_reasoning = F1_pred_reasoning / float(F1_count_reasoning)
            print("F1 SCORE:\t{:.4f}".format(F1_pred_reasoning / float(F1_count_reasoning)))
            print("BLEU SCORE:\t" + str(bleu_score))
            # print("PPL SCORE:\t{:.2f}".format(ppl_avg))
        elif args['dataset'] == 'babi':
            F1_score_multiwoz = F1_pred_babi / float(F1_count_babi)
            print("ACC SCORE:\t" + str(acc_score))
            print("F1 SCORE:\t{:.4f}".format(F1_pred_babi / float(F1_count_babi)))
            print("BLEU SCORE:\t" + str(bleu_score))
            # print("PPL SCORE:\t{:.2f}".format(ppl_avg))
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            # print("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))
            print("BLEU SCORE:\t" + str(bleu_score))
            # print("PPL SCORE:\t{:.2f}".format(ppl_avg))

        if (early_stop == 'BLEU'):
            # if (bleu_score >= matric_best):
            self.save_model('BLEU-' + str(bleu_score))
            print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            # if (F1_score >= matric_best):
            self.save_model('ENTF1-{:.4f}'.format(F1_score))
            print("MODEL SAVED")
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count
