import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from collections import OrderedDict
from learn_placing.training.utils import compute_rotation_matrix_from_ortho6d, RotRepr

def conv2D_outshape(in_shape, Cout, kernel, padding=(0,0), stride=(1,1), dilation=(1,1)):
    if len(in_shape)==2:
        Hin, Win = in_shape
    else:
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
        input_dim = [16,16], 
        kernel_sizes = [(5,5), (3,3)],
        cnn_out_channels = [32, 64],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 128,
        rnn_neurons = 128,
        rnn_layers = 2,
        fc_neurons = [128, 64],
        with_gripper = False,
        only_gripper = False,
        gripper_rot_type = RotRepr.quat,
        output_type = RotRepr.ortho6d
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
        self.output_type = output_type
        self.with_gripper = with_gripper
        self.only_gripper = only_gripper
        self.gripper_rot_type = gripper_rot_type

        assert output_type in ["quat", "ortho6d", "sincos"], f"unknown output type {output_type}"

        if self.output_type == RotRepr.quat:
            self.output_size = 4
        elif self.output_type == RotRepr.ortho6d:
            self.output_size = 6
        elif self.output_type == RotRepr.sincos:
            self.output_size = 2
        
        if self.gripper_rot_type == RotRepr.quat:
            self.gripper_input_size = 4

        # if there's no additional gripper input, set it additional dim to 0
        if not self.with_gripper: self.gripper_input_size = 0

        if not self.only_gripper:
            # input channels are whatever comes out of the previous layer.
            # first time it's the number of image dimensions
            self.cnn_in_channels = [1] + self.cnn_out_channels[:-1]

            # one conv for left, one for right sensor sequences
            self.conv1, self.rnn1 = self._conv_pre("left")
            self.conv2, self.rnn2 = self._conv_pre("right")

            # MLP marrying the two preprocessed input sequences
            self.mlpindim = 2*self.rnn_neurons+self.gripper_input_size
        else:
            self.mlpindim = self.gripper_input_size

        self.mlp = nn.Sequential(
            nn.Linear(self.mlpindim, self.fc_neurons[0]),
            nn.ReLU(),
            nn.Linear(self.fc_neurons[0], self.fc_neurons[1]),
            nn.ReLU(),
            nn.Linear(self.fc_neurons[1], self.output_size),
        )

    def forward(self, x: Tensor, gr: Tensor):
        """
        xs has dimensions: (batch, sensors, sequence, H, W)
        * we input the sequence of tactile images in the channels dimension
        * we have to pass each sensor sequence to their respective CNN-RNN pre-processors
        -> select xs[:,S,:], where S is the sensors index in {0,1}
        * then we loop over each image in the sequence, pass it into the CNN individually, concatenate the result and pass it into the RNN
        """
        if not isinstance(x, Tensor): x = torch.Tensor(x)

        if not self.only_gripper:
            cnnout1 = []
            cnnout2 = []
            for s in range(x.shape[2]): # loop over sequence frames
                """
                xs[:,0,s,:] is of shape (batch, H, W). the channel dimension is lost after we select it with `s`.
                however, we need to have the channel dimension for the conv layers (even though it will be of dimensionality one).
                -> we unsqueeze, yielding (batch, channel, H, W) with channel=1.
                """
                cnnout1.append(self.conv1(torch.unsqueeze(x[:,0,s,:], 1)))
                cnnout2.append(self.conv2(torch.unsqueeze(x[:,1,s,:], 1)))
            """
            * CNN output is a list of length SEQUENCE, each element with size (batch, self.conv_output).
            * stack makes this (sequence, batch, conv_out)
            * transpose swaps the first dimensions, arriving at a batch-first configuration: (batch, sequence, conv_out)

            fun fact: we only need to transpose here to be batch-frist conform, a mode that we explicitly need to set when creating the LSTM.
            the default mode is sequence-first, which would save us two transpose operations, but break consistency with other layers
            """
            cnnout1 = torch.stack(cnnout1).transpose_(0,1)
            cnnout2 = torch.stack(cnnout2).transpose_(0,1)
            
            rnnout1, (_, _) = self.rnn1(cnnout1, None)
            rnnout2, (_, _) = self.rnn2(cnnout2, None)

            mlpin = torch.cat([
                rnnout1[:,-1,:], 
                rnnout2[:,-1,:]
            ], axis=1)

            if self.with_gripper:
                mlpin = torch.cat([mlpin, gr], axis=1)
        else: # -> self.only_gripper == True
            mlpin = gr

        mlpout = self.mlp(mlpin)

        if self.output_type == RotRepr.ortho6d: mlpout = compute_rotation_matrix_from_ortho6d(mlpout)
        elif self.output_type == RotRepr.sincos: mlpout = torch.tanh(mlpout)

        return mlpout

    def _conv_pre(self, name):
        layers = []
        conv_outshape = None
        for i, (kern, inc, outc) in enumerate(zip(
                self.kernel_sizes, 
                self.cnn_in_channels, 
                self.cnn_out_channels
            )):

            layers.append((f"conv2d_{i}_{name}", nn.Conv2d(
                    in_channels=inc, 
                    out_channels=outc, 
                    kernel_size=kern, 
                    stride=self.conv_stride, 
                    padding=self.conv_padding)
            ))
            layers.append((f"batch_norm_{i}_{name}", nn.BatchNorm2d(outc, momentum=0.01)))
            layers.append((f"relu_conv_{i}_{name}", nn.ReLU(inplace=True)))# why inplace?
            conv_outshape = conv2D_outshape(
                self.input_dim if conv_outshape is None else conv_outshape,
                Cout=outc,
                padding=self.conv_padding,
                kernel=kern,
                stride=self.conv_stride
            )
        layers.append((f"flatten_{name}", nn.Flatten()))

        # TODO do we need this FC layer here or do we just pass the flattened conv output onwards?
        # the authors use two FC layers that they don't mention in the paper
        layers.append((f"post_cnn_linear_{name}", nn.Linear(np.prod(conv_outshape), self.conv_output)))
        layers.append((f"post_cnn_relu_{name}", nn.ReLU()))

        rnn = nn.LSTM(
            input_size=self.conv_output,
            hidden_size=self.rnn_neurons,
            num_layers=self.rnn_layers,
            batch_first=True
        )
        return nn.Sequential(OrderedDict(layers)), rnn

if __name__ == "__main__":
    import torch

    from learn_placing.common import load_dataset
    
    from torch.utils.tensorboard import SummaryWriter

    from learn_placing.training import TactileInsertionRLNet

    base_path = f"{__file__.replace(__file__.split('/')[-1], '')}"
    ds = load_dataset(f"{base_path}/test_samples")

    net = TactileInsertionRLNet()
    res = net(ds["tactile"])
    print(res.shape)

    # summary(net, input_size=(2, 40, 16, 16))
    
    # sw = SummaryWriter()
    # sw.add_graph(net, torch.randn((30, 2, 40, 16, 16)))
    # sw.close()
    
    pass