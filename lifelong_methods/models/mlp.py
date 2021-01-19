import torch
import torch.nn as nn

import experiments.utils


class MLP(nn.Module):
    # If multi_head is false, A single head is constructed with the total number of classes
    def __init__(self, input_shape, num_tasks, classes_per_task, num_hidden_layers=1, hidden_sizes=128,
                 multi_head=False):
        super(MLP, self).__init__()
        self.hidden_sizes = experiments.utils.extend_list(hidden_sizes, num_hidden_layers)
        self.classes_per_task = experiments.utils.extend_list(classes_per_task, num_tasks)
        self.num_hidden_layers = num_hidden_layers
        self.num_tasks = num_tasks
        self.multi_head = multi_head

        self.layers = nn.ModuleList([nn.Linear(input_shape, self.hidden_sizes[0], bias=True)])
        self.layers.extend(nn.ModuleList([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1], bias=True)
                                          for i in range(self.num_hidden_layers - 1)]))
        if self.multi_head:
            self.output_layer = nn.ModuleList(
                [nn.Linear(self.hidden_sizes[-1], num_classes) for num_classes in self.classes_per_task])
        else:
            self.output_layer = nn.Linear(self.hidden_sizes[-1], sum(self.classes_per_task))

        self.relu = torch.nn.ReLU()

    def forward(self, input_):
        x = self.layers[0](input_)
        x = self.relu(x)
        for layer in self.layers[1:]:
            x = layer(x)
            x = self.relu(x)
        if self.multi_head:
            output = [self.output_layer[t](x) for t in range(self.num_tasks)]
        else:
            output = self.output_layer(x)
        return output, x
