# During evaluation, this cell sets skip_training to True
# skip_training = True
import time
import tools, warnings
warnings.showwarning = tools.customwarn
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import tools

# When running on your own computer, you can specify the data directory by:
# data_dir = tools.select_data_dir('/your/local/data/directory')
data_dir = tools.select_data_dir()

# Select the device for training (use GPU if you have one)
#device = torch.device('cuda:0')
device = torch.device('cuda:0')

import data
trainset = data.Sudoku(data_dir, train=True)
testset = data.Sudoku(data_dir, train=False)

x, y = trainset[0]


def sudoku_to_labels(x):
    """Convert one-hot coded sudoku puzzles to labels. -1 corresponds to missing labels.

    Args:
      x of shape (n_rows=9, n_colums=9, n_digits=9): Tensor with a sudoku board. The digits are one-hot coded.
                  Cells with unknown digits have all zeros along the third dimension.
    """
    assert x.shape == torch.Size([9, 9, 9]), "Wrong shape {}".format(x.shape)
    is_filled = x.sum(dim=2)
    y = x.argmax(dim=2)
    y[~is_filled.bool()] = -1
    return y


def sudoku_edges():
    sudoko = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                       [9, 10, 11, 12, 13, 14, 15, 16, 17],
                       [18, 19, 20, 21, 22, 23, 24, 25, 26],
                       [27, 28, 29, 30, 31, 32, 33, 34, 35],
                       [36, 37, 38, 39, 40, 41, 42, 43, 44],
                       [45, 46, 47, 48, 49, 50, 51, 52, 53],
                       [54, 55, 56, 57, 58, 59, 60, 61, 62],
                       [63, 64, 65, 66, 67, 68, 69, 70, 71],
                       [72, 73, 74, 75, 76, 77, 78, 79, 80]])

    src_ids = []
    dst_ids = []

    for i in range(9):
        for j in range(9):
            sub_row = sudoko[i, :]
            sub_col = sudoko[:, j]

            current_value = list(set(sub_row).intersection(sub_col))[0]

            for k in range(9):
                row_ = sub_row[k]
                col_ = sub_col[k]

                if row_ != current_value:
                    src_ids.append(row_)
                    dst_ids.append(current_value)
                if col_ != current_value:
                    src_ids.append(col_)
                    dst_ids.append(current_value)

            c_x = (i // 3) * 3
            c_y = (j // 3) * 3

            for bi in range(3):
                for by in range(3):
                    value_x = sudoko[c_x + bi, c_y + by]
                    if (value_x not in sub_row and value_x not in sub_col):
                        src_ids.append(value_x)
                        dst_ids.append(current_value)

    src_ids = torch.tensor(src_ids, dtype=torch.long)
    dst_ids = torch.tensor(dst_ids, dtype=torch.long)

    return src_ids, dst_ids

sudoku_src_ids, sudoku_dst_ids = sudoku_edges()

def collate(list_of_samples):
    batch_size = len(list_of_samples)
    n_nodes = 81

    inputs = torch.zeros((batch_size * n_nodes, 9), dtype=torch.long)
    targets = torch.zeros((batch_size * n_nodes), dtype=torch.long)
    src_ids = torch.zeros(batch_size * 1620, dtype=torch.long)
    dst_ids = torch.zeros(batch_size * 1620, dtype=torch.long)

    i = 0
    add_value = torch.tensor([0], dtype=torch.long)
    added_val = torch.tensor([81], dtype=torch.long)

    for inp, tar in list_of_samples:
        inputs[i * n_nodes:(i + 1) * n_nodes, :] = inp
        targets[i * n_nodes:(i + 1) * n_nodes] = tar

        src_new = torch.add(sudoku_src_ids, add_value)
        tar_new = torch.add(sudoku_dst_ids, add_value)

        src_ids[i * 1620:(i + 1) * 1620] = src_new
        dst_ids[i * 1620:(i + 1) * 1620] = tar_new

        i += 1
        add_value = torch.add(add_value, added_val)

    inputs = inputs.to(device)
    targets = targets.to(device)
    src_ids = src_ids.to(device)
    dst_ids = dst_ids.to(device)

    return inputs, targets, src_ids, dst_ids

trainloader = DataLoader(trainset, batch_size=16, collate_fn=collate, shuffle=True)
testloader = DataLoader(testset, batch_size=16, collate_fn=collate, shuffle=False)


class GNN(nn.Module):
    def __init__(self, n_iters=7, n_node_features=10, n_node_inputs=9, n_edge_features=11, n_node_outputs=9):
        super(GNN, self).__init__()

        self.n_iters = n_iters
        self.n_node_features = n_node_features
        self.n_node_inputs = n_node_inputs
        self.n_edge_features = n_edge_features
        self.n_node_outputs = n_node_outputs

        self.output = nn.Linear(n_node_features, n_node_outputs)

        self.msg_net = nn.Sequential(nn.Linear(2 * n_node_features, 96),
                                     nn.ReLU(),
                                     nn.Linear(96, 96),
                                     nn.ReLU(),
                                     nn.Linear(96, self.n_edge_features)
                                     )

        self.act = nn.Softmax(dim=2)
        self.rnn = nn.GRU(n_edge_features + n_node_outputs, n_node_features)


    def forward(self, node_inputs, src_ids, dst_ids):
        """
        Args:
          node_inputs of shape (n_nodes, n_node_inputs): Tensor of inputs to every node of the graph.
          src_ids of shape (n_edges): Indices of source nodes of every edge.
          dst_ids of shape (n_edges): Indices of destination nodes of every edge.

        Returns:
          outputs of shape (n_iters, n_nodes, n_node_outputs): Outputs of all the nodes at every iteration of the
              graph neural network.
        """
        n_nodes = node_inputs.shape[0]
        self.n_edges = src_ids.shape[0]
        self.state = torch.zeros((1, n_nodes, self.n_node_features)).to(device)
        total_itr = torch.tensor((), dtype=torch.long).to(device)

        def state_to_msg(src_ids, dst_ids):
            msg_state_src = torch.zeros((self.n_edges, self.n_node_features)).to(device)
            msg_state_tar = torch.zeros((self.n_edges, self.n_node_features)).to(device)
            for k in range(self.n_edges):
                src_id = src_ids[k].item()
                dst_id = dst_ids[k].item()
                msg_state_src[k, :] = self.state[0, src_id, :]
                msg_state_tar[k, :] = self.state[0, dst_id, :]

            return torch.cat((msg_state_src, msg_state_tar), dim=1)

        for i in range(self.n_iters):
            msg_state = state_to_msg(src_ids, dst_ids)
            msgs = self.msg_net(msg_state)  # n_edges, n_edge_features

            index = dst_ids.type(torch.int32)
            agg_msgs = torch.zeros((n_nodes, self.n_edge_features)).to(device)
            agg_msgs = agg_msgs.index_add(0, index, msgs)

            gru_input = torch.cat((agg_msgs, node_inputs), dim=1).view(1, n_nodes, -1)
            output, h_n = self.rnn(gru_input, self.state)

            self.state = output
            out_iter = F.log_softmax(self.output(self.state.squeeze()), dim=1)
            out_iter = out_iter.reshape(1, n_nodes, -1)
            total_itr = torch.cat((total_itr, out_iter), 0)

        return self.act(total_itr)

# Create network
gnn = GNN()
gnn.to(device)

def fraction_of_solved_puzzles(gnn, testloader):
    with torch.no_grad():
        n_test = 0
        n_test_solved = 0
        for i, (inputs, targets, src_ids, dst_ids) in enumerate(testloader):
            # inputs is [n_nodes, 9*9, 9]
            # targets is [n_nodes]
            batch_size = inputs.size(0) // 81
            inputs, targets = inputs.to(device), targets.to(device)
            src_ids, dst_ids = src_ids.to(device), dst_ids.to(device)

            outputs = gnn(inputs, src_ids, dst_ids)  # [n_iters, batch*n_nodes, 9]
            solution = outputs.view(gnn.n_iters, batch_size, 9, 9, 9)

            final_solution = solution[-1].argmax(dim=3)
            solved = (final_solution.view(-1, 81) == targets.view(batch_size, 81)).all(dim=1)
            n_test += solved.size(0)
            n_test_solved += solved.sum().item()
            return n_test_solved / n_test


# Implement the training loop here
if not False:
    # YOUR CODE HERE
    # raise NotImplementedError()
    optimize = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    for epoch in range(30):
        episode_loss = []
        for batch_idx, batch in enumerate(trainloader):
            start = time.time()
            optimize.zero_grad()
            inputs, targets, src_ids, dst_ids = batch

            outputs = gnn(inputs, src_ids, dst_ids)
            losses = torch.tensor([0], dtype=torch.float32).to(device)

            end = time.time()
            print(start - end)

            for ite in range(gnn.n_iters):
                losses += loss(outputs[ite], targets)

            losses = losses * (1.0 / gnn.n_iters)
            episode_loss.append(losses.item())
            end = time.time()
            print(start - end)
            losses.backward()

            optimize.step()

        print("---------------")
        print("loss at epoch {} is {} with the largest being: {}".format(epoch, sum(episode_loss) / len(episode_loss),
                                                                         max(episode_loss)))
        results = fraction_of_solved_puzzles(gnn, testloader)

        print("result: ", results)

tools.save_model(gnn, '1_gnn.pth', confirm=True)