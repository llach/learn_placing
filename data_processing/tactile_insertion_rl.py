import numpy as np
import torch.nn as nn

from torch import Tensor, cat

def conv2D_outshape(img_size, padding, kernel, stride):

	output_shape=(np.floor((img_size[0] + 2 * padding[0] - (kernel[0] - 1) - 1) / stride[0] + 1).astype(int),
				  np.floor((img_size[1] + 2 * padding[1] - (kernel[1] - 1) - 1) / stride[1] + 1).astype(int))

	return output_shape


class TactileInsertionRLNet(nn.Module):

    """
    network code:
    https://github.com/siyuandong16/Tactile_insertion_with_RL/blob/master/supervised_learning/crnn_model.py

    notes on the code:

    * they define dropout layers and parametrize them in the constructor, but don't use it (see the forward() functions). this is consistent with the paper

    """

    def __init__(self, 
        input_dim = [40,16,16], 
        kernel_sizes = [(5,5), (3,3)],
        cnn_out_channels = [32, 64],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 512,
        rnn_neurons = 512,
        rnn_layers = 2,
        fc_neurons = [512, 256],
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
        self.cnn_in_channels = [len(self.input_dim)] + self.cnn_out_channels[:-1]

        # one conv for left, one for right sensor sequences
        self.conv1 = self._conv_pre()
        self.conv2 = self._conv_pre()

        # MLP marrying the two preprocessed input sequences
        self.mlp = nn.Sequential(
            nn.Linear(2*self.rnn_neurons, self.fc_neurons[0]),
            nn.ReLU(),
            nn.Linear(self.fc_neurons[0], self.fc_neurons[1]),
            nn.ReLU(),
            nn.Linear(self.fc_neurons[1], self.output_size),
        )

    def forward(self, xs: list[Tensor]):
        return self.mlp(cat([
            self.conv1(xs[0]), 
            self.conv1(xs[1])
        ]))

    def _conv_pre(self):
        layers = []
        conv_outshape = None
        for kern, inc, outc in zip(
                self.kernel_sizes, 
                self.cnn_in_channels, 
                self.cnn_out_channels
            ):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=inc, 
                        out_channels=outc, 
                        kernel_size=kern, 
                        stride=self.conv_stride, 
                        padding=self.conv_padding
                    ),
                    nn.BatchNorm2d(outc, momentum=0.01),
                    nn.ReLU(inplace=True),   # ?
                )
            )
            self.conv_outshape = conv2D_outshape(
                self.input_dim if conv_outshape is None else conv_outshape,
                padding=self.conv_padding,
                kernel=kern,
                stride=self.conv_stride
            )
        layers.append(nn.Linear(self.cnn_out_channels[-1]*np.prod(self.conv_outshape), self.conv_output))
        # do we need a ReLu here?
        layers.append(nn.ReLU())
        layers.append(nn.LSTM(
            input_size=self.conv_output,
            hidden_size=self.rnn_neurons,
            num_layers=self.rnn_layers,
            batch_first=True
        ))
        return nn.Sequential(*layers)
        