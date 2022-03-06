import torch
import torch.nn as nn
from utils.config import *
import pdb


class ProofGenerator(nn.Module):
    def __init__(self, emb_size, hidden_size, relations_cnt, entities_cnt, rel_emb, ent_emb):
        super(ProofGenerator, self).__init__()
        self.name = "ProofGenerator"

        self.emb_size = emb_size
        self.rel_num = relations_cnt
        self.ent_num = entities_cnt
        self.hidden_size = hidden_size
        self.shared_rel_emb = rel_emb  # nb_relations*embed_dim
        self.shared_ent_emb = ent_emb  # nb_entities*embed_dim

        if USE_CUDA:
            self.projections2rel = [nn.Linear(emb_size, emb_size).cuda() for _ in range(2)]
        else:
            self.projections2rel = [nn.Linear(emb_size, emb_size) for _ in range(2)]
        self.projections2entity = nn.Linear(3*emb_size, emb_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, left_child, right_child, query):
        batch_size = query.shape[0]

        left_child_rel = self.projections2rel[0](query[:, 1, :])  # left_child_rel: batch_size*embed_dim
        right_child_rel = self.projections2rel[1](query[:, 1, :])  # right_child_rel: batch_size*embed_dim
        query_concat = torch.cat([query[:, 0, :], query[:, 1, :], query[:, 2, :]], dim=1)
        query4entity = self.projections2entity(query_concat)  # query4entity: batch_size*emb_dim

        rel_emb = self.shared_rel_emb.weight.data
        ent_emb = self.shared_ent_emb.weight.data

        rel_emb_expand = rel_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # batch_size*nb_relations*embed_dim
        ent_emb_expand = ent_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # batch_size*nb_entities*embed_dim

        query4entity_expand = query4entity.unsqueeze(1).repeat(1, self.ent_num, 1)  # batch_size*nb_entities*embed_dim
        left_child_rel_expand = left_child_rel.unsqueeze(1).repeat(1, self.rel_num, 1)  # batch_size*nb_relations*embed_dim
        right_child_rel_expand = right_child_rel.unsqueeze(1).repeat(1, self.rel_num, 1)  # batch_size*nb_relations*embed_dim

        prob_logits = torch.sum(query4entity_expand * ent_emb_expand, 2)  # batch_size*nb_entities
        prob_ = self.softmax(prob_logits)  # batch_size*nb_entities
        prob = prob_.unsqueeze(2).repeat(1, 1, self.emb_size)  # batch_size*nb_entities*embed_dim
        o_ent = torch.sum(prob * ent_emb_expand, 1)  # batch_size*embed_dim

        prob_logits = torch.sum(left_child_rel_expand * rel_emb_expand, 2)  # batch_size*nb_relations
        prob_ = self.softmax(prob_logits)  # batch_size*nb_relations
        prob = prob_.unsqueeze(2).repeat(1, 1, self.emb_size)  # batch_size*nb_relations*embed_dim
        o_left_r = torch.sum(prob * rel_emb_expand, 1)  # batch_size*embed_dim

        prob_logits = torch.sum(right_child_rel_expand * rel_emb_expand, 2)  # batch_size*nb_relations
        prob_ = self.softmax(prob_logits)  # batch_size*nb_relations
        prob = prob_.unsqueeze(2).repeat(1, 1, self.emb_size)  # batch_size*nb_relations*embed_dim
        o_right_r = torch.sum(prob * rel_emb_expand, 1)  # batch_size*embed_dim

        left_child = torch.stack([left_child[:, 0, :], o_left_r, o_ent], dim=1)
        right_child = torch.stack([o_ent, o_right_r, right_child[:, 2, :]], dim=1)  # bug here, fixed!!! right_child[:, 0, :]--->right_child[:, 2, :]
        return left_child, right_child
