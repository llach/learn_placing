import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from collections import OrderedDict
from learn_placing.training.utils import RotRepr, compute_rotation_matrix_from_quaternion

def conv2D_outshape(in_shape, Cout, kernel, padding=(0,0), stride=(1,1), dilation=(1,1)):
    if len(in_shape)==2:
        Hin, Win = in_shape
    else:
        _, Hin, Win = in_shape

    Hout = np.int8(np.floor(((Hin+2*padding[0]-dilation[0]*(kernel[0]-1)-1)/(stride[0]))+1))
    Wout = np.int8(np.floor(((Win+2*padding[1]-dilation[1]*(kernel[1]-1)-1)/(stride[1]))+1))
    return (Cout, Hout, Wout)

def conv3D_outshape(in_shape, Cout, kernel, padding=(0,0,0), stride=(1,1,1), dilation=(1,1,1)):
    if len(in_shape)==3:
        Din, Hin, Win = in_shape
    else:
        _, Din, Hin, Win = in_shape

    Dout = np.int(np.floor(((Din+2*padding[0]-dilation[0]*(kernel[0]-1)-1)/(stride[0]))+1))
    Hout = np.int(np.floor(((Hin+2*padding[1]-dilation[1]*(kernel[1]-1)-1)/(stride[1]))+1))
    Wout = np.int(np.floor(((Win+2*padding[2]-dilation[2]*(kernel[2]-1)-1)/(stride[2]))+1))
    return (Cout, Dout, Hout, Wout)

class MyrmexNet(nn.Module):
    
    def __init__(self, 
        input_dim = [16,16],
        kernel_sizes = [(5,5), (3,3)],
        cnn_out_channels = [32, 64],
        conv_stride = (2,2),
        conv_padding = (0,0),
        conv_output = 128,
        fc_neurons = [128, 64],
        with_tactile = True,
        with_gripper = False,
        with_ft = False,
        output_type = RotRepr.ortho6d
        ) -> None:
        super().__init__()
        assert np.any([with_tactile, with_gripper, with_ft]), "no input modality"

        self.input_dim = input_dim
        self.kernel_sizes = kernel_sizes
        self.cnn_out_channels = cnn_out_channels
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding
        self.conv_output = conv_output
        self.fc_neurons = fc_neurons

        self.with_tactile = with_tactile
        self.with_gripper = with_gripper
        self.with_ft = with_ft

        self.gripper_input_size = 4 if self.with_gripper else 0
        self.ft_input_size = 6 if self.with_ft else 0
        self.tactile_input_size = 0

        self.output_type = output_type

        if self.output_type == RotRepr.angle:
            self.output_size = 1
        elif self.output_type == RotRepr.ortho6d:
            self.output_size = 6
    
        if self.with_tactile:
            self.cnn_in_channels = [2] + self.cnn_out_channels[:-1]
            self.conv = self._conv_pre()
            self.tactile_input_size = self.conv_output

        self.mlpindim = sum([self.gripper_input_size, self.tactile_input_size, self.ft_input_size])
        self.mlp = nn.Sequential(
            nn.Linear(self.mlpindim, self.fc_neurons[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_neurons[0], self.fc_neurons[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.fc_neurons[1], self.output_size),
        )

    def forward(self, x: Tensor, gr: Tensor, ft: Tensor):
        """
        xs [batch, 1, H, W]  single, merged Myrmex frame
        gr [batch, 4]        quaternion world to gripper, Qwg
        ft [batch, 6]        single F/T sensor reading
        """
        if not isinstance(x, Tensor) and x is not None: x = torch.Tensor(x)

        mlp_inputs = []
        if self.with_tactile: mlp_inputs.append(self.conv(x))
        if self.with_gripper: mlp_inputs.append(gr)
        if self.with_ft:  mlp_inputs.append(ft)

        if len(mlp_inputs)>1:
            mlpin = torch.cat(mlp_inputs, axis=1)
        else: 
            mlpin = mlp_inputs[0]

        mlpout = self.mlp(mlpin)

        if self.output_type == RotRepr.ortho6d: mlpout = compute_rotation_matrix_from_quaternion(mlpout)
        return mlpout

    def _conv_pre(self):
        layers = []
        conv_outshape = None
        for i, (kern, inc, outc) in enumerate(zip(
                self.kernel_sizes,
                self.cnn_in_channels,
                self.cnn_out_channels
            )):

            layers.append((f"conv2d_{i}", nn.Conv2d(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=kern,
                    stride=self.conv_stride,
                    padding=self.conv_padding)
            ))
            # layers.append((f"batch_norm_{i}_{name}", nn.BatchNorm2d(outc, momentum=0.01)))
            layers.append((f"relu_conv_{i}", nn.ReLU(inplace=True)))# why inplace?
            conv_outshape = conv2D_outshape(
                self.input_dim if conv_outshape is None else conv_outshape,
                Cout=outc,
                padding=self.conv_padding,
                kernel=kern,
                stride=self.conv_stride
            )
        layers.append((f"flatten", nn.Flatten()))

        layers.append((f"post_cnn_linear", nn.Linear(np.prod(conv_outshape), self.conv_output)))
        layers.append((f"post_cnn_relu", nn.ReLU()))

        return nn.Sequential(OrderedDict(layers))

