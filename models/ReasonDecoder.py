import torch
import torch.nn as nn
from utils.utils_general import _cuda
from utils.config import *


class ReasonDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, dropout):
        super(ReasonDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.C = shared_emb
        self.softmax = nn.Softmax(dim=1)
        self.rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.projector = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, encode_hidden, question_generator, reasoner, target_batches, max_target_length, batch_size, use_teacher_forcing, get_decoded_words, context_arr, context_arr_lengths, facts_arr, ent_index, global_pointer, conv_arr_plain, response_plain, kb_arr_plain):
        # Initialize variables for vocab and pointer
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_candidate_prob = _cuda(torch.zeros(max_target_length, batch_size, context_arr.shape[1]))
        all_triples_label = _cuda(torch.zeros(max_target_length, batch_size, args['max_neg_cnt']+1))
        all_triples_scores = _cuda(torch.zeros(max_target_length, batch_size, args['max_neg_cnt']+1))
        memory_mask_for_step = _cuda(torch.ones((context_arr.shape[0], context_arr.shape[1])))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        decoded_fine, decoded_coarse = [], []

        hidden = self.relu(self.projector(encode_hidden)).unsqueeze(0)
        # if batch_size == 1:
        #     hidden = hidden.unsqueeze(0)

        # Start to generate word-by-word
        for t in range(max_target_length):
            embed_q = self.dropout_layer(self.C(decoder_input))  # b * e
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            _, hidden = self.rnn(embed_q.unsqueeze(0), hidden)

            p_vocab = self.attend_vocab(self.C.weight, hidden.squeeze(0))
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)  # topvi: [batch_size * 1]

            # query question generator and reasoner using hidden state
            structure_type_logits, structure_type_action, structure_type_loss, \
            query_entity_h_logits, query_entity_h_action, query_entity_h_loss, \
            query_entity_t_logits, query_entity_t_action, query_entity_t_loss, candidates_prob, candidates_prob_logits = question_generator(context_arr, context_arr_lengths, hidden.squeeze(0), global_pointer, True)
            all_candidate_prob[t] = candidates_prob_logits

            question_arr, structure_type_action, pos = self.decode_questions(structure_type_action, query_entity_h_action,
                                                                        query_entity_t_action, candidates_prob,
                                                                        context_arr, memory_mask_for_step)
            # convert queston_arr to id
            question_arr_id = self.convert_questions_to_id(question_arr)  # batch_size * (max_neg_cnt+1) * 3
            if USE_CUDA:
                question_arr_id = torch.Tensor(question_arr_id).cuda()
            else:
                question_arr_id = torch.Tensor(question_arr_id)

            scores = reasoner(question_arr_id, facts_arr)
            all_triples_scores[t] = scores

            # Calc BCELoss labels
            bce_labels = self.calc_bce_labels(question_arr, ent_index, structure_type_action, target_batches[:, t])
            all_triples_label[t] = bce_labels

            # 2.Get reasoning outputs
            extracted_entities = self.extract_max_confidence_conclusions(question_arr_id, structure_type_action, batch_size)  # [batch_size * (max_neg_cnt+1)]

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            if get_decoded_words:
                search_len = args['max_neg_cnt']+1
                temp_f, temp_c = [], []
                for bi in range(batch_size):
                    token = topvi[bi].item()  # topvi[:,0][bi].item()
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if int(extracted_entities[:, i][bi].item()) != self.lang.entity2index["$$$$"]:
                                cw = self.lang.index2entity[int(extracted_entities[:, i][bi].item())]
                                break
                        temp_f.append(cw)
                        if args['record']:
                            memory_mask_for_step[bi, int(pos[bi, i].item())] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])
                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, decoded_fine, decoded_coarse, all_candidate_prob, all_triples_scores, all_triples_label

    def decode_questions(self, structure_type, query_entity_h_prob, query_entity_t_prob, candidates_prob, context_arr, memory_mask_for_step):
        candidates_prob = candidates_prob * memory_mask_for_step
        _, ca = candidates_prob.topk((args['max_neg_cnt']+1), dim=1, largest=True, sorted=True)  # batch_size * (max_neg_cnt+1)

        question_arr = []  # batch_size * (max_neg_cnt+1) * 3
        # map one-hot to index
        batch_size = context_arr.shape[0]
        structure_type_idx = torch.arange(0, 2).unsqueeze(0).repeat(batch_size, 1)
        query_entity_h_idx = torch.arange(0, self.lang.n_entities).unsqueeze(0).repeat(batch_size, 1)
        query_entity_t_idx = torch.arange(0, self.lang.n_relations).unsqueeze(0).repeat(batch_size, 1)
        if USE_CUDA:
            structure_type_idx = structure_type_idx.cuda()
            query_entity_h_idx = query_entity_h_idx.cuda()
            query_entity_t_idx = query_entity_t_idx.cuda()
        structure_type_action = torch.sum(structure_type * structure_type_idx, dim=1)
        query_entity_h_action = torch.sum(query_entity_h_prob * query_entity_h_idx, dim=1)
        query_entity_t_action = torch.sum(query_entity_t_prob * query_entity_t_idx, dim=1)
        for bi, t in enumerate(structure_type_action):
            question_ent = self.lang.index2entity[query_entity_h_action[bi].item()]
            question_rel = self.lang.index2relation[query_entity_t_action[bi].item()]
            conclusion_arr_bi = []
            for i in range(args['max_neg_cnt']+1):
                candidate_ent = context_arr[bi][ca[bi][i]][0]
                candidate_ent_token = self.lang.index2word[candidate_ent.item()]
                if t.item() == 0:
                    candidate_conclusion = [candidate_ent_token, question_rel, question_ent]
                elif t.item() == 1:
                    candidate_conclusion = [question_ent, question_rel, candidate_ent_token]
                else:
                    candidate_conclusion = [candidate_ent_token, question_rel, question_ent]
                conclusion_arr_bi.append(candidate_conclusion)
            question_arr.append(conclusion_arr_bi)

        return question_arr, structure_type_action, ca

    def convert_questions_to_id(self, question_arr):  # batch_size * (max_neg_cnt+1) * 3
        batch_size = len(question_arr)
        question_arr_ret = []
        for bi in range(batch_size):
            conclusion_arr = []
            for i in range(args['max_neg_cnt']+1):
                if question_arr[bi][i][0] not in self.lang.entity2index:
                    h = self.lang.entity2index["UNK"]
                else:
                    h = self.lang.entity2index[question_arr[bi][i][0]]
                if question_arr[bi][i][1] not in self.lang.relation2index:
                    r = self.lang.relation2index["UNK"]
                else:
                    r = self.lang.relation2index[question_arr[bi][i][1]]
                if question_arr[bi][i][2] not in self.lang.entity2index:
                    t = self.lang.entity2index["UNK"]
                else:
                    t = self.lang.entity2index[question_arr[bi][i][2]]
                conclusion_arr.append([h, r, t])
            question_arr_ret.append(conclusion_arr)
        return question_arr_ret

    def extract_max_confidence_conclusions(self, question_arr_id, structure_type_action, batch_size):
        max_confidence_entities = []
        for bt in range(batch_size):
            entitites = []
            for i in range(args['max_neg_cnt']+1):
                if structure_type_action[bt].item() == 0:
                    ent = question_arr_id[bt][i][0]
                elif structure_type_action[bt].item() == 1:
                    ent = question_arr_id[bt][i][2]
                else:
                    ent = question_arr_id[bt][i][0]
                entitites.append(ent)
            max_confidence_entities.append(entitites)
        if USE_CUDA:
            max_confidence_entities = torch.Tensor(max_confidence_entities).cuda()
        else:
            max_confidence_entities = torch.Tensor(max_confidence_entities)
        return max_confidence_entities

    def calc_bce_labels(self, question_arr, gold_ent, structure_type, target_batch):
        batch_size = structure_type.shape[0]
        batch_labels = []
        for bt in range(batch_size):
            labels = []
            for i in range(args['max_neg_cnt']+1):
                predicted_triple = question_arr[bt][i]
                if structure_type[bt].item() == 0:
                    predicted_ent = predicted_triple[0]
                elif structure_type[bt].item() == 1:
                    predicted_ent = predicted_triple[2]
                else:
                    predicted_ent = predicted_triple[0]
                if predicted_ent in gold_ent[bt]:
                    label = 1
                else:
                    label = 0
                labels.append(label)
            batch_labels.append(labels)
        batch_labels = torch.Tensor(batch_labels).float()
        if USE_CUDA:
            batch_labels = batch_labels.cuda()
        return batch_labels

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        return scores_
