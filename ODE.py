from layer import dilated_inception, mixprop, CGP, graph_constructor
import torchdiffeq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ODEFunc(nn.Module):
    def __init__(self, stnet):
        super(ODEFunc, self).__init__()
        self.stnet = stnet
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        x = self.stnet(x)
        return x


class ODEBlock(nn.Module):
    def __init__(self, odefunc, method, step_size, rtol, atol, adjoint=False, perturb=False):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
        self.adjoint = adjoint
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)
        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                             method=self.method,
                                             options=dict(step_size=self.step_size, perturb=self.perturb))
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=dict(step_size=self.step_size, perturb=self.perturb))

        return out[-1]


class STBlock(nn.Module):

    def __init__(self, receptive_field, dilation, hidden_channels, dropout, method, time, step_size, alpha,
                 rtol, atol, adjoint, perturb=False):
        super(STBlock, self).__init__()
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.graph = None
        self.dropout = dropout
        self.new_dilation = 1
        self.dilation_factor = dilation
        self.inception_1 = dilated_inception(hidden_channels, hidden_channels, dilation_factor=1)
        self.inception_2 = dilated_inception(hidden_channels, hidden_channels, dilation_factor=1)
        self.gconv_1 = CGP(hidden_channels, hidden_channels, alpha=alpha,
                           method=method, time=time, step_size=step_size, rtol=rtol, atol=atol,
                           adjoint=adjoint, perturb=perturb)
        self.gconv_2 = CGP(hidden_channels, hidden_channels, alpha=alpha,
                           method=method, time=time, step_size=step_size, rtol=rtol, atol=atol,
                           adjoint=adjoint, perturb=perturb)

    def forward(self, x):
        x = x[..., -self.intermediate_seq_len:]
        for tconv in self.inception_1.tconv:
            tconv.dilation = (1, self.new_dilation)
        for tconv in self.inception_2.tconv:
            tconv.dilation = (1, self.new_dilation)

        filter = self.inception_1(x)
        filter = torch.tanh(filter)
        gate = self.inception_2(x)
        gate = torch.sigmoid(gate)
        x = filter * gate

        self.new_dilation *= self.dilation_factor
        self.intermediate_seq_len = x.size(3)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gconv_1(x, self.graph) + self.gconv_2(x, self.graph.transpose(1, 0))

        x = nn.functional.pad(x, (self.receptive_field - x.size(3), 0))

        return x

    def setGraph(self, graph):
        self.graph = graph

    def setIntermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field

