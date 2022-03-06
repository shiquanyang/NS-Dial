import torch
import torch.nn as nn
from models.ProofGenerator import ProofGenerator
from models.GaussianKernel import GaussianKernel
from utils.config import *
import json
import copy


class ReasonEngine(nn.Module):
    def __init__(self, max_depth, emb_size, hidden_size, relations_cnt, entities_cnt, lang):
        super(ReasonEngine, self).__init__()
        self.name = "ReasonEngine"

        self.max_depth = max_depth
        self.embed_size = emb_size
        self.lang = lang

        self.entity_embeddings = nn.Embedding(entities_cnt, emb_size)
        self.relation_embeddings = nn.Embedding(relations_cnt, emb_size)
        self.entity_embeddings.weight.data.normal_(0, 0.1)
        self.relation_embeddings.weight.data.normal_(0, 0.1)

        self.proof_generator = ProofGenerator(emb_size, hidden_size, relations_cnt, entities_cnt, self.relation_embeddings, self.entity_embeddings)
        self.kernel = GaussianKernel(emb_size)

    def parse_nodes(self, nodes, depth, parsed_nodes, is_leaf, visit_trace_node, visit_trace, branch):
        if depth == 0:
            parsed_nodes.append(nodes)
            is_leaf.append(1)
            if branch == 'root':
                visit_trace_node_t = copy.deepcopy(visit_trace_node)
            elif branch == 'left':
                visit_trace_node_t = copy.deepcopy(visit_trace_node)
                visit_trace_node_t.append(0)
            elif branch == 'right':
                visit_trace_node_t = copy.deepcopy(visit_trace_node)
                visit_trace_node_t.append(1)
            visit_trace.append(visit_trace_node_t)
            return
        parsed_nodes.append(nodes[0])
        is_leaf.append(0)
        if branch == 'root':
            visit_trace_node_t = copy.deepcopy(visit_trace_node)
        elif branch == 'left':
            visit_trace_node_t = copy.deepcopy(visit_trace_node)
            visit_trace_node_t.append(0)
        elif branch == 'right':
            visit_trace_node_t = copy.deepcopy(visit_trace_node)
            visit_trace_node_t.append(1)
        visit_trace.append(visit_trace_node_t)
        self.parse_nodes(nodes[1], depth-1, parsed_nodes, is_leaf, visit_trace_node_t, visit_trace, 'left')
        self.parse_nodes(nodes[2], depth-1, parsed_nodes, is_leaf, visit_trace_node_t, visit_trace, 'right')

    def get_score_by_visit_trace(self, scores, visit_trace, depth, bt):
        if depth == 1:
            return scores[visit_trace[0]][bt]
        ret = self.get_score_by_visit_trace(scores[visit_trace[0]], visit_trace[1:], depth-1, bt)
        return ret

    def show_trees(self, nodes_trace, scores_trace):
        entity_embeddings = self.entity_embeddings.weight.data
        relation_embeddings = self.relation_embeddings.weight.data

        all_depths_tokens_trees, all_depths_scores = [], []
        batch_size = nodes_trace[0][0].shape[0]
        for bt in range(batch_size):
            tokens, scores_ = [], []
            for depth, nodes in enumerate(nodes_trace):
                depth = depth + 1
                parsed_nodes, is_leaf, visit_trace, visit_trace_node = [], [], [], []
                self.parse_nodes(nodes, depth, parsed_nodes, is_leaf, visit_trace_node, visit_trace, 'root')
                all_tokens_trees, all_scores = [], []
                for idx, triples in enumerate(parsed_nodes):
                    tokens_tree, leaf_scores = [], []
                    for pos, word in enumerate(triples[bt]):
                        if pos == 0 or pos == 2:
                            score = self.kernel.pairwise(word, entity_embeddings)
                            token = self.lang.index2entity[score.argmax().item()]
                            tokens_tree.append(token)
                        elif pos == 1:
                            score = self.kernel.pairwise(word, relation_embeddings)
                            token = self.lang.index2relation[score.argmax().item()]
                            tokens_tree.append(token)
                    if is_leaf[idx]:
                        score_visit_trace = visit_trace[idx]
                        if depth == 0:
                            scores_trace_t = scores_trace[0][bt]
                        else:
                            scores_trace_t = self.get_score_by_visit_trace(scores_trace[depth - 1], score_visit_trace, len(score_visit_trace), bt)
                        leaf_scores.append(scores_trace_t.item())
                    elif is_leaf[idx] == 0:
                        leaf_scores.append(-1)
                    all_tokens_trees.append(tokens_tree)
                    all_scores.append(leaf_scores)
                tokens.append(all_tokens_trees)
                scores_.append(all_scores)
            all_depths_tokens_trees.append(tokens)
            all_depths_scores.append(scores_)

        return all_depths_tokens_trees, all_depths_scores

    def save_trees(self, all_depths_tokens_trees, all_depths_scores):
        fout = open('/home/shiquan/Projects/DialogueReasoning/outputs/multiwoz_output_proof_trees.json', 'w')
        tree_dict = {}
        for bt, batch_tokens in enumerate(all_depths_tokens_trees):
            if bt not in tree_dict:
                tree_dict[bt] = {}
            for depth, nodes in enumerate(batch_tokens):
                depth = depth + 1
                if depth not in tree_dict[bt]:
                    tree_dict[bt][depth] = {}
                for node_id, tokens in enumerate(nodes):
                    if node_id not in tree_dict[bt][depth]:
                        tree_dict[bt][depth][node_id] = {}
                    if 'triple' not in tree_dict[bt][depth][node_id]:
                        tree_dict[bt][depth][node_id]['triple'] = []
                    tree_dict[bt][depth][node_id]['triple'].append(tokens[0])
                    tree_dict[bt][depth][node_id]['triple'].append(tokens[1])
                    tree_dict[bt][depth][node_id]['triple'].append(tokens[2])
                    if 'score' not in tree_dict[bt][depth][node_id]:
                        tree_dict[bt][depth][node_id]['score'] = []
                    tree_dict[bt][depth][node_id]['score'].append(all_depths_scores[bt][depth - 1][node_id][0])

        proof_trees = json.dumps(tree_dict, ensure_ascii=False, indent=1)
        fout.write(proof_trees)
        fout.close()

    def reasoning_chain_planning(self, query):
        batch_size = query.shape[0]
        if USE_CUDA:
            left_child = torch.stack([query[:, 0, :], torch.zeros(batch_size, self.embed_size).cuda(), torch.zeros(batch_size, self.embed_size).cuda()], dim=1)
            right_child = torch.stack([torch.zeros(batch_size, self.embed_size).cuda(), torch.zeros(batch_size, self.embed_size).cuda(), query[:, 2, :]], dim=1)
        else:
            left_child = torch.stack([query[:, 0, :], torch.zeros(batch_size, self.embed_size), torch.zeros(batch_size, self.embed_size)], dim=1)
            right_child = torch.stack([torch.zeros(batch_size, self.embed_size), torch.zeros(batch_size, self.embed_size), query[:, 2, :]], dim=1)
        return left_child, right_child

    def reasoning_chain_realization(self, left_child, right_child, query):
        left_child, right_child = self.proof_generator(left_child, right_child, query)
        return left_child, right_child

    def generate_child_terms(self, query):
        left_child, right_child = self.reasoning_chain_planning(query)
        left_child, right_child = self.reasoning_chain_realization(left_child, right_child, query)
        return left_child, right_child

    def generate_proof_tree(self, query, facts, depth):
        batch_size = query.shape[0]

        if depth == 0:
            scores = []
            for bt in range(batch_size):
                score = self.kernel(query[bt], facts[bt])
                scores.append(score)
            batch_scores = torch.stack(scores, dim=0).squeeze(dim=1)
            res, _ = torch.max(batch_scores, dim=1, keepdim=True)
            return query, res

        nodes, scores = [], []
        left_child, right_child = self.generate_child_terms(query)
        left_child, scores_left = self.generate_proof_tree(left_child, facts, depth-1)
        right_child, scores_right = self.generate_proof_tree(right_child, facts, depth-1)

        nodes.append(query)
        nodes.append(left_child)
        nodes.append(right_child)
        scores.append(scores_left)
        scores.append(scores_right)

        return nodes, scores

    def decode_scores(self, scores, depth, parsed_scores):
        if depth == 0:
            parsed_scores.append(scores)
            return
        self.decode_scores(scores[0], depth-1, parsed_scores)
        self.decode_scores(scores[1], depth-1, parsed_scores)

    def parse_all_leaf_scores(self, scores, max_depth):
        parsed_scores = []
        self.decode_scores(scores, max_depth, parsed_scores)
        return parsed_scores

    def forward(self, query, facts):
        batch_size = query.shape[0]
        nb_query = query.shape[1]

        facts_expand = facts.unsqueeze(1).repeat(1, nb_query, 1, 1)

        q_s, q_r, q_o = query[:, :, 0].unsqueeze(2), query[:, :, 1].unsqueeze(2), query[:, :, 2].unsqueeze(2)
        f_s, f_r, f_o = facts_expand[:, :, :, 0].unsqueeze(3), facts_expand[:, :, :, 1].unsqueeze(3), facts_expand[:, :, :, 2].unsqueeze(3)

        q_s_emb = self.entity_embeddings(q_s.contiguous().view(q_s.size(0), -1).long())
        q_s_emb = q_s_emb.view(q_s.size() + (q_s_emb.size(-1),))

        q_r_emb = self.relation_embeddings(q_r.contiguous().view(q_r.size(0), -1).long())
        q_r_emb = q_r_emb.view(q_r.size() + (q_r_emb.size(-1),))

        q_o_emb = self.entity_embeddings(q_o.contiguous().view(q_o.size(0), -1).long())
        q_o_emb = q_o_emb.view(q_o.size() + (q_o_emb.size(-1),))

        f_s_emb = self.entity_embeddings(f_s.contiguous().view(f_s.size(0), -1).long())
        f_s_emb = f_s_emb.view(f_s.size() + (f_s_emb.size(-1),))

        f_r_emb = self.relation_embeddings(f_r.contiguous().view(f_r.size(0), -1).long())
        f_r_emb = f_r_emb.view(f_r.size() + (f_r_emb.size(-1),))

        f_o_emb = self.entity_embeddings(f_o.contiguous().view(f_o.size(0), -1).long())
        f_o_emb = f_o_emb.view(f_o.size() + (f_o_emb.size(-1),))

        query_emb = torch.stack([q_s_emb, q_r_emb, q_o_emb], dim=2).squeeze(3)
        facts_emb = torch.stack([f_s_emb, f_r_emb, f_o_emb], dim=3).squeeze(4)

        # query_emb = self.embeddings(query.contiguous().view(query.size(0), -1).long())  # query: batch_size*3
        # query_emb = query_emb.view(query.size() + (query_emb.size(-1),))  # query_emb: batch_size*3*embed_dim
        # facts_emb = self.embeddings(facts_expand.contiguous().view(facts_expand.size(0), -1).long())  # facts: batch_size*nb_facts*3
        # facts_emb = facts_emb.view(facts_expand.size() + (facts_emb.size(-1),))  # facts_emb: batch_size*nb_facts*3*embed_dim

        query_emb_t = query_emb.view(-1, query_emb.shape[2], query_emb.shape[3])
        facts_emb_t = facts_emb.view(-1, facts_emb.shape[2], facts_emb.shape[3], facts_emb.shape[4])

        # res = [self.generate_proof_tree(query_emb_t, facts_emb_t, depth) for depth in range(1, self.max_depth+1)]  # generate proof tree for each depth.
        res = [self.generate_proof_tree(query_emb_t, facts_emb_t, 1)]

        proof_scores, nodes_trace, scores_trace = [], [], []

        for depth, content in enumerate(res):
            depth = depth + 1
            nodes, scores = content[0], content[1]
            nodes_trace.append(nodes)
            scores_trace.append(scores)

            parsed_scores = self.parse_all_leaf_scores(scores, depth)  # parse all leaf node scores.
            stacked_scores = torch.stack(parsed_scores, dim=2).squeeze(dim=1)  # stack all leaf node scores to compute min.
            proof_score, _ = torch.min(stacked_scores, dim=1, keepdim=True)

            proof_scores.append(proof_score)  # save per-depth proof score.

        stack_proof_scores = torch.stack(proof_scores, dim=2).squeeze(dim=1)  # stack all depth scores to compute max.
        p_scores_t, _ = torch.max(stack_proof_scores, dim=1, keepdim=True)

        p_scores = p_scores_t.view(batch_size, -1)

        if args['show_trees']:
            token_trees, score_trees = self.show_trees(nodes_trace, scores_trace)
            self.save_trees(token_trees, score_trees)

        return p_scores
