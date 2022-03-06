import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "UNK"}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

        self.type2index = {}
        self.index2type = {PAD_token: "PAD", UNK_token: "UNK", 2: "PRE", 3: "NEXT"}
        self.n_types = len(self.index2type)
        self.type2index = dict([(v, k) for k, v in self.index2type.items()])
        self.type2cnt = {}

        self.entity2index, self.relation2index = {}, {}
        self.index2entity, self.index2relation = {UNK_token: "UNK"}, {UNK_token: "UNK"}
        self.n_entities = len(self.index2entity)
        self.n_relations = len(self.index2relation)
        self.entity2index = dict([(v, k) for k, v in self.index2entity.items()])
        self.relation2index = dict([(v, k) for k, v in self.index2relation.items()])

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def index_entities(self, kb_arr, type):
        for triple in kb_arr:
            for idx, token in enumerate(triple):
                if type == "ent":
                    self.index_entity(idx, token)
                elif type == "rel":
                    self.index_relation(idx, token)

    def index_entity(self, idx, token):
        if token not in self.entity2index and (idx == 0 or idx == 2):
            self.entity2index[token] = self.n_entities
            self.index2entity[self.n_entities] = token
            self.n_entities += 1

    def index_relation(self, idx, token):
        if token not in self.relation2index and (idx == 1):
            self.relation2index[token] = self.n_relations
            self.index2relation[self.n_relations] = token
            self.n_relations += 1

    def index_type(self, type):
        for direction in type:
            for type_triple in direction:
                for type in type_triple:
                    if type not in self.type2index:
                        self.type2index[type] = self.n_types
                        self.index2type[self.n_types] = type
                        self.n_types += 1
                    if type not in self.type2cnt:
                        self.type2cnt[type] = 1
                    else:
                        self.type2cnt[type] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, ent2id, rel2id):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.ent2id = ent2id
        self.rel2id = rel2id

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        conclusion_arr = self.data_info['conclusion_arr'][index]
        conclusion_arr = self.preprocess_ent_rel(conclusion_arr, self.ent2id, self.rel2id)
        facts_arr = self.data_info['facts_arr'][index]
        facts_arr = self.preprocess_ent_rel(facts_arr, self.ent2id, self.rel2id)
        conclusion_label_arr = torch.Tensor(self.data_info['conclusion_label_arr'][index])
        structure_type = torch.Tensor(self.data_info['structure_type'][index])
        query_entity_h = self.data_info['query_entity_h'][index]
        query_entity_h = self.preprocess_query_ent(query_entity_h, self.ent2id)
        query_entity_t = self.data_info['query_entity_t'][index]
        query_entity_t = self.preprocess_query_rel(query_entity_t, self.rel2id)
        candidates_pointer = torch.Tensor(self.data_info['candidates_pointer'][index])

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def preprocess_query_ent(self, sequence, ent2id):
        story = []
        for word in sequence:
            temp = ent2id[word] if word in ent2id else UNK_token
            story.append(temp)
        story = torch.Tensor(story)
        return story

    def preprocess_query_rel(self, sequence, rel2id):
        story = []
        for word in sequence:
            temp = rel2id[word] if word in rel2id else UNK_token
            story.append(temp)
        story = torch.Tensor(story)
        return story

    def preprocess_ent_rel(self, sequence, ent2id, rel2id):
        story = []
        for i, word_triple in enumerate(sequence):
            story.append([])
            for ii, word in enumerate(word_triple):
                if ii == 0 or ii == 2:
                    temp = ent2id[word] if word in ent2id else UNK_token
                elif ii == 1:
                    temp = rel2id[word] if word in rel2id else UNK_token
                story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def preprocess_conclusion(self, sequence, word2id):
        """Converts words to ids."""
        story = [word2id[word] if word in word2id else UNK_token for word in sequence]
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_conclusion(sequences):
            lengths = [len(seq) for seq in sequences]
            loss_lengths = [1 for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            padded_seqs = torch.ones(len(sequences), max_len, 3).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, loss_lengths

        def merge_facts(sequences):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            padded_seqs = torch.ones(len(sequences), max_len, 3).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                if len(seq) != 0:
                    padded_seqs[i, :end, :] = seq[:end]
            return padded_seqs, lengths

        def merge_label(sequences):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            padded_seqs = torch.ones(len(sequences), max_len).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        ptr_index, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)
        conclusion_arr, conclusion_arr_lengths = merge_conclusion(item_info['conclusion_arr'])
        facts_arr, facts_arr_lengths = merge_facts(item_info['facts_arr'])
        conclusion_label_arr, _ = merge_label(item_info['conclusion_label_arr'])
        structure_type, structure_type_lengths = merge(item_info['structure_type'], False)
        query_entity_h, query_entity_h_lengths = merge(item_info['query_entity_h'], False)
        query_entity_t, query_entity_t_lengths = merge(item_info['query_entity_t'], False)
        candidates_pointer, candidates_pointer_lengths = merge(item_info['candidates_pointer'], False)

        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        if (len(list(kb_arr.size())) > 1): kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())
        conclusion_arr = _cuda(conclusion_arr.contiguous())
        facts_arr = _cuda(facts_arr.contiguous())
        conclusion_label_arr = _cuda(conclusion_label_arr.contiguous())
        structure_type = _cuda(structure_type.contiguous())
        query_entity_h = _cuda(query_entity_h.contiguous())
        query_entity_t = _cuda(query_entity_t.contiguous())
        candidates_pointer = _cuda(candidates_pointer.contiguous())

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths
        data_info['conclusion_arr_lengths'] = conclusion_arr_lengths
        data_info['structure_type_lengths'] = structure_type_lengths
        data_info['query_entity_h_lengths'] = query_entity_h_lengths
        data_info['query_entity_t_lengths'] = query_entity_t_lengths
        data_info['candidates_pointer_lengths'] = candidates_pointer_lengths

        return data_info


def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if (type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_entities(pair['kb_arr'], "ent")
            lang.index_entities(pair['kb_arr'], "rel")

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang.entity2index, lang.relation2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              # shuffle = False,
                                              collate_fn=dataset.collate_fn)
    return data_loader


def compute_dataset_length(data_length, batch_size):
    return int(data_length / batch_size)