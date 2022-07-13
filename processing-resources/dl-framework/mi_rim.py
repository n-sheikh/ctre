import torch
import torch.nn as nn
import numpy as np
import math
from nn_utils import Embedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)


# noinspection PyUnresolvedReferences
class GroupLinearLayer(nn.Module):
    def __init__(self, d_in, d_out, num_mechanisms):
        super(GroupLinearLayer, self).__init__()
        self.w = nn.init.xavier_normal_(nn.Parameter(torch.randn(num_mechanisms, d_in, d_out)))

    def forward(self, x):
        x = torch.bmm(x, self.w)
        return x


class InputSelection(torch.nn.Module):
    """
    A module to help the mechanism choose its input. Less the attention
    on the input, less the inclination to choose the input.
    """

    def __init__(self, input_size, hidden_size, query_size, key_size, val_size):
        super(InputSelection, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.query_layer = nn.Linear(hidden_size, query_size)
        self.key_layer = nn.Linear(input_size, key_size)
        self.val_layer = nn.Linear(input_size, val_size)

    def forward(self, input, h_prev):
        augmented_input = torch.cat((torch.zeros(1, self.input_size, device=device), input), dim=0)
        query = self.query_layer(h_prev)
        key = self.key_layer(augmented_input)
        value = self.val_layer(augmented_input)
        att_score = torch.softmax(torch.matmul(query, torch.transpose(key, 1, 0)) / math.sqrt(self.key_size), dim=1)
        selected_input = torch.matmul(att_score, value)
        return att_score, selected_input


class Mechanism(torch.nn.Module):
    """
    The module representing each mechanism. Currently, we only support rnn's and lstms as encoders.
    """
    def __init__(self, rnn_type, input_size, hidden_size, input_q_size, input_key_size, input_val_size):
        super(Mechanism, self).__init__()
        self.input_selection = InputSelection(input_size, hidden_size, input_q_size, input_key_size, input_val_size)
        self.rnn_type = rnn_type

        if rnn_type == 'vanilla':
            self.rnn = nn.RNNCell(input_size, hidden_size)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTMCell(input_size, hidden_size)

    def recurrence(self, x_t, h_t, c_t):
        if self.rnn_type == 'vanilla':
            return self.rnn(x_t, h_t)
        elif self.rnn_type == 'lstm':
            return self.rnn(x_t, (h_t, c_t))


class MI_RIM(torch.nn.Module):
    def __init__(self, rnn_type, num_mech, num_active, hidden_size, input_sizes):
        super(MI_RIM, self).__init__()
        self.num_mechanisms = num_mech
        self.num_active = num_active
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.mechanisms = nn.ModuleList([Mechanism(self.rnn_type, input_sizes[i], hidden_size, input_q_size=256,
                                                   input_key_size=256, input_val_size=input_sizes[i]) for i in
                                         range(num_mech)])
        self.inter_query_layer = GroupLinearLayer(hidden_size, hidden_size, self.num_mechanisms)
        self.inter_key_layer = GroupLinearLayer(hidden_size, hidden_size, self.num_mechanisms)
        self.inter_val_layer = GroupLinearLayer(hidden_size, hidden_size, self.num_mechanisms)
        self.intitial_hiddens = torch.zeros(self.num_mechanisms, 1, self.hidden_size, device=device)

    def interaction(self, hiddens, mask):
        query = self.inter_query_layer(hiddens)
        key = self.inter_key_layer(hiddens)
        value = self.inter_val_layer(hiddens)
        att_scores = torch.softmax(torch.matmul(query.squeeze(), torch.transpose(key.squeeze(), 0, 1)), dim=1)
        att_scores = torch.bmm(mask, att_scores.view(self.num_mechanisms, 1, -1))
        new_hiddens = torch.matmul(att_scores.squeeze(), value.squeeze()).view(self.num_mechanisms, 1, -1)
        return new_hiddens

    def forward(self, inputs):
        seq_len = len(inputs[0])
        prev_hiddens = self.intitial_hiddens
        if self.rnn_type == 'lstm':
            prev_cells = torch.zeros(self.num_mechanisms, 1, self.hidden_size, device=device)
        all_hiddens = list()
        all_activation = list()
        for t in range(seq_len):
            temp_cells = torch.zeros(self.num_mechanisms, 1, self.hidden_size, device=device)
            selection_att_scores = torch.empty(self.num_mechanisms, 2, device=device)
            selected_inputs = list()
            for mech_indx in range(self.num_mechanisms):
                att_score, selected_input = self.mechanisms[mech_indx].input_selection(
                    inputs[mech_indx][t, :].view(1, -1), prev_hiddens[mech_indx])
                selection_att_scores[mech_indx] = att_score
                selected_inputs.append(selected_input)

            active_idx = torch.topk(selection_att_scores[:, 0], self.num_active, largest=False).indices

            all_activation.append(active_idx)

            mask = torch.zeros(1, self.num_mechanisms, device=device)
            mask[:, active_idx] = 1
            mask = mask.view(self.num_mechanisms, 1, 1)

            temp_hiddens = torch.bmm((1 - mask), prev_hiddens)

            for mech_indx in active_idx:
                if self.rnn_type == 'vanilla':
                    temp_hidden = self.mechanisms[mech_indx].recurrence(selected_inputs[mech_indx],
                                                                        prev_hiddens[mech_indx, :], None)

                elif self.rnn_type == 'lstm':
                    temp_hidden, new_cell = self.mechanisms[mech_indx].recurrence(selected_inputs[mech_indx],
                                                                                  prev_hiddens[mech_indx, :],
                                                                                  prev_cells[mech_indx, :])
                    temp_cells[mech_indx] = new_cell
                temp_hiddens[mech_indx] = temp_hidden

            new_hiddens = self.interaction(temp_hiddens, mask) + torch.bmm((1 - mask), prev_hiddens) \
                          + torch.bmm(mask, temp_hiddens)
            all_hiddens.append(new_hiddens.view(1, self.num_mechanisms * self.hidden_size))
            prev_hiddens = new_hiddens
            prev_cells = temp_cells

        return torch.stack(all_hiddens, dim=0).squeeze(), all_activation
