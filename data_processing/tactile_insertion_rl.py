import numpy as np
import torch.nn as nn

from torch import Tensor, cat
from collections import OrderedDict

def conv2D_outshape(in_shape, Cout, kernel, padding=(0,0), stride=(1,1), dilation=(1,1)):
    _, Hin, Win = in_shape

    Hout = np.int(np.floor(((Hin+2*padding[0]-dilation[0]*(kernel[0]-1)-1)/(stride[0]))+1))
    Wout = np.int(np.floor(((Win+2*padding[1]-dilation[1]*(kernel[1]-1)-1)/(stride[1]))+1))
    return (Cout, Hout, Wout)


class TactileInsertionRLNet(nn.Module):

    """
    network code:
    https://github.com/siyuandong16/Tactile_insertion_with_RL/blob/master/supervised_learning/crnn_model.py

    notes on their code:

    * they define dropout layers and parametrize them in the constructor, but don't use it (see the forward() functions). this is consistent with the paper

    deviations:
    * we ony use 2 conv layers since our input has lower dimensionality than theirs
    * the FC layer between conv and LSTM is 128 instead of 512 neurons since the flattened conv output is already 256 units
    * consequently, our MLP is smaller, too

    """

    def __init__(self, 
        input_dim = [40,16,16], 
        kernel_sizes = [(5,5), (3,3)],
        cnn_out_channels = [32, 64],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 128,
        rnn_neurons = 128,
        rnn_layers = 2,
        fc_neurons = [128, 64],
        output_size = 3
        ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.kernel_sizes = kernel_sizes
        self.cnn_out_channels = cnn_out_channels
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_output = conv_output
        self.rnn_neurons = rnn_neurons
        self.rnn_layers = rnn_layers
        self.fc_neurons = fc_neurons
        self.output_size = output_size

        # input channels are whatever comes out of the previous layer.
        # first time it's the number of image dimensions
        self.cnn_in_channels = [self.input_dim[0]] + self.cnn_out_channels[:-1]

        # one conv for left, one for right sensor sequences
        self.conv1 = self._conv_pre("left")
        self.conv2 = self._conv_pre("right")

        # MLP marrying the two preprocessed input sequences
        self.mlp = nn.Sequential(
            nn.Linear(2*self.rnn_neurons, self.fc_neurons[0]),
            nn.ReLU(),
            nn.Linear(self.fc_neurons[0], self.fc_neurons[1]),
            nn.ReLU(),
            nn.Linear(self.fc_neurons[1], self.output_size),
        )

    def forward(self, xs: list[Tensor]):
        """
        xs has dimensions: (batch, sensors, channels, H, W)
        we input the sequence of tactile images in the channels dimension
        we have to pass each sensor sequence to their respective CNN-LSTM pre-processors
        -> select xs[:,S,:], where S is the sensors index in {0,1}
        """
        # TODO MLP and LSTM still missing :(
        return cat([
            self.conv1(xs[:,0,:]), 
            self.conv1(xs[:,1,:])
        ])

    def _conv_pre(self, name):
        layers = []
        conv_outshape = None
        for i, (kern, inc, outc) in enumerate(zip(
                self.kernel_sizes, 
                self.cnn_in_channels, 
                self.cnn_out_channels
            )):
            layers.append(
                nn.Sequential(OrderedDict([
                    (f"conv2d_{i}_{name}", nn.Conv2d(
                        in_channels=inc, 
                        out_channels=outc, 
                        kernel_size=kern, 
                        stride=self.conv_stride, 
                        padding=self.conv_padding
                    )),
                    (f"batch_norm_{i}_{name}", nn.BatchNorm2d(outc, momentum=0.01)),
                    (f"relu_conv_{i}_{name}", nn.ReLU(inplace=True)),   # why inplace?
                ]))
            )
            conv_outshape = conv2D_outshape(
                self.input_dim if conv_outshape is None else conv_outshape,
                Cout=outc,
                padding=self.conv_padding,
                kernel=kern,
                stride=self.conv_stride
            )
        layers.append(nn.Flatten())

        # TODO do we need this FC layer here or do we just pass the flattened conv output onwards?
        layers.append(nn.Linear(np.prod(conv_outshape), self.conv_output))
        layers.append(nn.ReLU())

        # TODO no idea what to put as sequence length ...
        # layers.append(nn.LSTM(
        #     input_size=self.conv_output,
        #     hidden_size=self.rnn_neurons,
        #     num_layers=self.rnn_layers,
        #     batch_first=True
        # ))
        return nn.Sequential(*layers)