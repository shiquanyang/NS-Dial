import torch
import torch.nn as nn
from utils.config import *
from utils.utils_general import _cuda


class MultiTaskQuestionGeneratorForDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, vocab, embedding_dim, lang, hops, n_layers=1):
        super(MultiTaskQuestionGeneratorForDecoder, self).__init__()
        self.name = "MultiTaskQuestionGeneratorForDecoder"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.lang = lang
        self.max_hops = hops

        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(3*hidden_size, hidden_size)
        # StructurePredictor Head.
        self.W1 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        # QueryPredictor Head.
        self.query_predictor = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.W3 = nn.Linear(hidden_size, lang.n_entities)
        self.W4 = nn.Linear(hidden_size, lang.n_relations)
        # CandidatePredictor Head.
        for hop in range(self.max_hops+1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.candidate_predictor = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
        self.candidate_predictor.weight.data.normal_(0, 0.1)
        self.sigmoid = nn.Sigmoid()

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, context_arr, context_arr_lengths, hidden, global_pointer, is_decoding):
        # Shared Layer
        hidden = self.W(hidden).unsqueeze(0)

        # Task-specific Heads
        # - StructureTypePredictor Head
        structure_type_logits, structure_type_action, structure_type_loss = self.compute_structure_type(hidden.squeeze(0))

        # - QueryContentPredictor Head
        query_entity_h_logits, query_entity_h_action, query_entity_h_loss, \
            query_entity_t_logits, query_entity_t_action, query_entity_t_loss = self.compute_query_entities(hidden.squeeze(0))

        # - CandidatesPredictor Head
        candidates_prob, candidates_prob_logits = self.compute_candidates_probability(context_arr, context_arr_lengths, hidden.squeeze(0), global_pointer, is_decoding)

        return structure_type_logits, structure_type_action, structure_type_loss, \
               query_entity_h_logits, query_entity_h_action, query_entity_h_loss, \
               query_entity_t_logits, query_entity_t_action, query_entity_t_loss, candidates_prob, candidates_prob_logits

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        return scores_

    def compute_structure_type(self, hidden):
        sp_h = self.W1(hidden)  # [batch_size, 2]
        # sample an action
        sp_h_action = torch.nn.functional.gumbel_softmax(logits=sp_h, tau=1.0, hard=True)
        loss = None
        return sp_h, sp_h_action, loss

    def compute_query_entities(self, hidden):
        query_entity_h = self.W3(hidden)  # [batch_size, nb_entities]
        # sample an action
        query_entity_h_action = torch.nn.functional.gumbel_softmax(logits=query_entity_h, tau=1.0, hard=True)
        query_entity_h_loss = None

        query_entity_t = self.W4(hidden)  # [batch_size, nb_relations]
        # sample an action
        query_entity_t_action = torch.nn.functional.gumbel_softmax(logits=query_entity_t, tau=1.0, hard=True)
        query_entity_t_loss = None

        return query_entity_h, query_entity_h_action, query_entity_h_loss, query_entity_t, query_entity_t_action, query_entity_t_loss

    def compute_candidates_probability(self, story, context_arr_lengths, hidden, global_pointer, is_decoding):
        # Forward multiple hop mechanism
        # story = story.transpose(0, 1)
        story_size = story.size()
        u = [hidden]

        if not is_decoding:
            self.m_story = []
            for hop in range(self.max_hops):
                embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  # .long()) # b * (m * s) * e
                embed_A = embed_A.view(story_size + (embed_A.size(-1),))  # b * m * s * e
                embed_A = torch.sum(embed_A, 2).squeeze(2)  # b * m * e
                embed_A = self.dropout_layer(embed_A)

                if (len(list(u[-1].size())) == 1):
                    u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
                u_temp = u[-1].unsqueeze(1).expand_as(embed_A)
                prob_logit = torch.sum(embed_A * u_temp, 2)
                prob_ = self.softmax(prob_logit)

                embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
                embed_C = embed_C.view(story_size+(embed_C.size(-1),))
                embed_C = torch.sum(embed_C, 2).squeeze(2)

                prob = prob_.unsqueeze(2).expand_as(embed_C)
                o_k = torch.sum(embed_C*prob, 1)
                u_k = u[-1] + o_k
                u.append(u_k)
                self.m_story.append(embed_A)
            self.m_story.append(embed_C)
            ret = self.sigmoid(prob_logit)

            # Mask
            batch_size = len(context_arr_lengths)
            max_len = max(context_arr_lengths)
            lengths_tensor = torch.Tensor(context_arr_lengths).unsqueeze(1).repeat(1, max_len)
            comparison_tensor = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1)
            mask = torch.lt(comparison_tensor, lengths_tensor)
            if USE_CUDA:
                mask = mask.cuda()
            dummy_scores = torch.zeros_like(ret)
            masked_candidates_prob = torch.where(mask, ret, dummy_scores)
            return masked_candidates_prob
        else:
            for hop in range(self.max_hops):
                m_A = self.m_story[hop]
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
                if (len(list(u[-1].size())) == 1):
                    u[-1] = u[-1].unsqueeze(0)  ## used for bsz = 1.
                u_temp = u[-1].unsqueeze(1).expand_as(m_A)
                prob_logits = torch.sum(m_A * u_temp, 2)

                # Mask
                batch_size = len(context_arr_lengths)
                max_len = max(context_arr_lengths)
                lengths_tensor = torch.Tensor(context_arr_lengths).unsqueeze(1).repeat(1, max_len)
                comparison_tensor = torch.arange(0, max_len).unsqueeze(0).repeat(batch_size, 1)
                mask = torch.lt(comparison_tensor, lengths_tensor)
                if USE_CUDA:
                    mask = mask.cuda()
                dummy_scores = torch.ones_like(prob_logits) * -99999.0
                masked_candidates_prob = torch.where(mask, prob_logits, dummy_scores)

                prob_soft = self.softmax(masked_candidates_prob)
                m_C = self.m_story[hop + 1]
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
                prob = prob_soft.unsqueeze(2).expand_as(m_C)
                o_k = torch.sum(m_C * prob, 1)
                u_k = u[-1] + o_k
                u.append(u_k)
            return prob_soft, masked_candidates_prob


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))