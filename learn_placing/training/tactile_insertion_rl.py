import torch
import numpy as np
import torch.nn as nn

from enum import Enum
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

def conv3D_outshape(in_shape, Cout, kernel, padding=(0,0,0), stride=(1,1,1), dilation=(1,1,1)):
    if len(in_shape)==3:
        Din, Hin, Win = in_shape
    else:
        _, Din, Hin, Win = in_shape

    Dout = np.int(np.floor(((Din+2*padding[0]-dilation[0]*(kernel[0]-1)-1)/(stride[0]))+1))
    Hout = np.int(np.floor(((Hin+2*padding[1]-dilation[1]*(kernel[1]-1)-1)/(stride[1]))+1))
    Wout = np.int(np.floor(((Win+2*padding[2]-dilation[2]*(kernel[2]-1)-1)/(stride[2]))+1))
    return (Cout, Dout, Hout, Wout)


class ConvProc(str, Enum):
    TRL = "trl"
    SINGLETRL = "SINGLETRL"
    ONEFRAMESINGLETRL = "ONEFRAMESINGLETRL"
    TDCONV = "3DConv"

class TactilePlacingNet(nn.Module):
    

    """
    inspired from TactileInsertionNet with significant adaptations to our task:
    https://github.com/siyuandong16/Tactile_insertion_with_RL/blob/master/supervised_learning/crnn_model.py
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
        ft_rnn_neurons = 16,
        ft_rnn_layers = 2,
        fc_neurons = [128, 64],
        with_tactile = True,
        with_gripper = False,
        with_ft = False,
        gripper_rot_type = RotRepr.quat,
        output_type = RotRepr.ortho6d,
        preproc_type = ConvProc.TRL
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
        self.ft_rnn_neurons = ft_rnn_neurons
        self.ft_rnn_layers = ft_rnn_layers
        self.fc_neurons = fc_neurons
        self.output_type = output_type
        self.with_tactile = with_tactile
        self.with_gripper = with_gripper
        self.with_ft = with_ft
        self.gripper_rot_type = gripper_rot_type
        self.preproc_type = preproc_type

        assert np.any([with_tactile, with_gripper, with_ft]), "no input modality"
        assert output_type in ["quat", "ortho6d", "sincos"], f"unknown output type {output_type}"

        if self.output_type == RotRepr.quat:
            self.output_size = 4
        elif self.output_type == RotRepr.ortho6d:
            self.output_size = 6
        elif self.output_type == RotRepr.sincos:
            self.output_size = 2

        self.gripper_input_size = 4 if self.with_gripper else 0
        self.ft_input_size = self.ft_rnn_neurons if self.with_ft else 0
        self.tactile_input_size = 0
    
        if self.with_tactile:
            if self.preproc_type == ConvProc.TRL:
                # input channels are whatever comes out of the previous layer.
                # first time it's the number of image dimensions
                self.cnn_in_channels = [1] + self.cnn_out_channels[:-1]

                # one conv for left, one for right sensor sequences
                self.conv1, self.rnn1 = self._conv_pre("left")
                self.conv2, self.rnn2 = self._conv_pre("right")

                self.self.tactile_input_size  = 2*self.rnn_neurons
            elif self.preproc_type == ConvProc.SINGLETRL:
                self.cnn_in_channels = [2] + self.cnn_out_channels[:-1]

                self.conv, self.rnn = self._conv_pre("conv_proc")

                self.tactile_input_size = self.rnn_neurons

            elif self.preproc_type == ConvProc.ONEFRAMESINGLETRL:
                self.cnn_in_channels = [2] + self.cnn_out_channels[:-1]
                self.conv = self._oneframe_conv_pre("oneframe_conv_proc")
                self.tactile_input_size = self.conv_output

            elif self.preproc_type == ConvProc.TDCONV:
                self.cnn_in_channels = [2] + self.cnn_out_channels[:-1]

                self.conv, self.rnn, self.last_flatten_layer = self._conv_pre_3D("conv_proc_3D")

                self.tactile_input_size = self.rnn_neurons

        if self.with_ft:
            self.ftrnn = nn.LSTM(
                input_size=6,
                hidden_size=self.ft_rnn_neurons,
                num_layers=self.ft_rnn_layers,
                batch_first=True
            )

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
        xs has dimensions: (batch, sensors, sequence, H, W)
        * we input the sequence of tactile images in the channels dimension
        * we have to pass each sensor sequence to their respective CNN-RNN pre-processors
        -> select xs[:,S,:], where S is the sensors index in {0,1}
        * then we loop over each image in the sequence, pass it into the CNN individually, concatenate the result and pass it into the RNN

        gr has dimensions [batch, 4] and contains the world to gripper transformations represented as quaternion

        ft has dimensions [batch, 6] and contains raw FT sensor readings
        """
        if not isinstance(x, Tensor) and x is not None: x = torch.Tensor(x)

        mlp_inputs = []
        if self.with_tactile:
            if self.preproc_type == ConvProc.TRL:
                tacout = self.trl_proc(x, gr)
            elif self.preproc_type == ConvProc.SINGLETRL:
                tacout = self.single_trl_proc(x, gr)
            elif self.preproc_type == ConvProc.ONEFRAMESINGLETRL:
                tacout = self.one_frame_single_trl_proc(x, gr)
            elif self.preproc_type == ConvProc.TDCONV:
                tacout = self.threed_trl_proc(x, gr)
            mlp_inputs.append(tacout)

        if self.with_gripper:
            mlp_inputs.append(gr)

        if self.with_ft:
            # LSTM input shape: (batch, seq, M)
            # !! LSTM needs to have `batch_first` set to `True` during instanciation !!
            ftrnnout, (_,_) = self.ftrnn(ft, None)
            mlp_inputs.append(ftrnnout[:,-1,:])

        if len(mlp_inputs)>1:
            mlpin = torch.cat(mlp_inputs, axis=1)
        else: 
            mlpin = mlp_inputs[0]

        mlpout = self.mlp(mlpin)

        if self.output_type == RotRepr.ortho6d: mlpout = compute_rotation_matrix_from_ortho6d(mlpout)
        elif self.output_type == RotRepr.sincos: mlpout = torch.tanh(mlpout)

        ## TODO insert trafo multiplication

        return mlpout

    def single_trl_proc(self, x: Tensor, gr: Tensor):
        """ x has shape [batch,sensor,sequence,H,W]
        """
        cnnout = []
        for s in range(x.shape[2]):
            cnnout.append(self.conv(x[:,:,s,:,:]))

        # cnnout to [batch,sequence,cnn_out_neurons]
        cnnout = torch.stack(cnnout).transpose_(0,1)
        rnnout, (_, _) = self.rnn(cnnout, None)
        return rnnout[:,-1,:]

    def one_frame_single_trl_proc(self, x: Tensor, gr: Tensor):
        """ x has shape [batch,sensor,sequence,H,W]
            compare with above - we just bruteforcely select 10th frame
        """
        cnnout = self.conv(x[:,:,10,:,:])
        return cnnout

    def threed_trl_proc(self, x: Tensor, gr: Tensor):
        """ x has shape [batch,sensor,sequence,H,W]
        """
        threed_cnnout = self.conv(x)

        # cnnout to [batch,sequence,cnn_out_neurons]
        threed_cnnout = self.last_flatten_layer((threed_cnnout).transpose(1,2))

        rnnout, (_, _) = self.rnn(threed_cnnout, None)
        return rnnout[:,-1,:]


    def trl_proc(self, x: Tensor, gr: Tensor):
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

        return torch.cat([
            rnnout1[:,-1,:], 
            rnnout2[:,-1,:]
        ], axis=1)

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


    def _oneframe_conv_pre(self, name):
        # same as above, only without using any rnn
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
            # layers.append((f"batch_norm_{i}_{name}", nn.BatchNorm2d(outc, momentum=0.01)))
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

        return nn.Sequential(OrderedDict(layers))


    def _conv_pre_3D(self, name):
        layers = []
        conv_outshape = None
        for i, (kern, inc, outc) in enumerate(zip(
                self.kernel_sizes,
                self.cnn_in_channels,
                self.cnn_out_channels
            )):

            layers.append((f"conv2d_{i}_{name}", nn.Conv3d(
                    in_channels=inc,
                    out_channels=outc,
                    kernel_size=kern,
                    stride=self.conv_stride,
                    padding=self.conv_padding)
            ))
            # layers.append((f"batch_norm_{i}_{name}", nn.BatchNorm2d(outc, momentum=0.01)))
            layers.append((f"relu_conv_{i}_{name}", nn.ReLU(inplace=True)))# why inplace?
            conv_outshape = conv3D_outshape(
                self.input_dim if conv_outshape is None else conv_outshape,
                Cout=outc,
                padding=self.conv_padding,
                kernel=kern,
                stride=self.conv_stride
            )
            print (conv_outshape)
        layers.append((f"flatten_{name}", nn.Flatten(-2,-1)))

        # TODO do we need this FC layer here or do we just pass the flattened conv output onwards?
        # the authors use two FC layers that they don't mention in the paper
        layers.append((f"post_cnn_linear_{name}", nn.Linear(conv_outshape[-2]*conv_outshape[-1], self.conv_output)))
        self.conv_output = conv_outshape[0]*self.conv_output
        layers.append((f"post_cnn_relu_{name}", nn.ReLU()))

        rnn = nn.LSTM(
            input_size=self.conv_output,
            hidden_size=self.rnn_neurons,
            num_layers=self.rnn_layers,
            batch_first=True
        )

        last_layer = nn.Flatten(-2,-1)

        return nn.Sequential(OrderedDict(layers)), rnn, last_layer

