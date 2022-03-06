import torch
import torch.nn as nn
from utils.config import *
from utils.utils_general import _cuda
import random
# from transformers.tokenization_bert import BertTokenizer
# from transformers.modeling_bert import BertModel


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, vocab, embedding_dim, lang, n_layers=1):
        super(Encoder, self).__init__()
        self.name = "Encoder"

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.lang = lang

        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.W = nn.Linear(2*hidden_size, hidden_size)
        # self.W = nn.Linear(768, hidden_size)
        # self.tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL,
        #                                                do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
        # self.bert = BertModel.from_pretrained(BERT_PRETRAINED_MODEL)

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def get_state(self, bsz):
        """Get cell states and hidden states."""
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def trunc_seq(self, tokens, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
        l = 0
        r = len(tokens)
        trunc_tokens = list(tokens)
        while r - l > max_num_tokens:
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                l += 1
            else:
                r -= 1
        return trunc_tokens[l:r]

    def compute_bert_input(self, conv_arr_plain, batch_size):
        # convert to id and padding
        bert_input = [" ".join(elm) for elm in conv_arr_plain]
        lens = [len(ele.split(" ")) for ele in bert_input]
        max_len = max(lens)
        padded_seqs = torch.zeros(batch_size, min(max_len, MAX_INPUT_LEN)).long()
        input_mask = torch.zeros(batch_size, min(max_len, MAX_INPUT_LEN))
        for i, seq in enumerate(bert_input):
            end = min(lens[i], MAX_INPUT_LEN)
            tokens = self.tokenizer.tokenize(seq)
            tokens = self.trunc_seq(tokens, 512)
            seq_id = self.tokenizer.convert_tokens_to_ids(tokens)
            padded_seqs[i, :end] = torch.Tensor(seq_id[:end])
            input_mask[i, :end] = 1
        return padded_seqs, input_mask

    def forward(self, input_seqs, input_lengths):
        # - Encode Dialogue History
        # BERT Encoder
        # batch_size = input_seqs.size(0)
        # input_ids, input_mask = self.compute_bert_input(input_seqs, batch_size)
        # _, pooled_output = self.bert(input_ids, attention_mask=input_mask)
        # encoded_hidden = self.W(pooled_output)
        embedded = self.embedding(input_seqs.contiguous().view(input_seqs.size(0), -1).long())
        embedded = embedded.view(input_seqs.size()+(embedded.size(-1),))
        embedded = torch.sum(embedded, 2).squeeze(2)
        embedded = self.dropout_layer(embedded)
        hidden = self.get_state(input_seqs.size(1))
        # print("input_seqs out size: ", input_seqs.size())
        # print("embedded size: ", embedded.size())
        if input_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)
        if input_lengths:
           outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)
        hidden = self.W(torch.cat((hidden[0], hidden[1]), dim=1)).unsqueeze(0)
        # outputs = self.W(outputs)

        return hidden.squeeze(0)
