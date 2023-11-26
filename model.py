from torch.nn import BatchNorm2d, Conv2d, Parameter

from ODE import *
from utils import GCNPool, multi_gcn, Static_Transmit  # H_GCN_cpu
from utils import ST_BLOCK_0  # ASTGCN
from utils import ST_BLOCK_4  # STGCN
from utils import ST_BLOCK_5  # GRCN
from utils import ST_BLOCK_6  # OTSGGCN
from utils import Transmit
from utils import gate

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class ASTGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(ASTGCN,self).__init__()
        self.block1=ST_BLOCK_0(in_dim,dilation_channels,num_nodes,length,K,Kt)
        self.block2=ST_BLOCK_0(dilation_channels,dilation_channels,num_nodes,length,K,Kt)
        self.final_conv=Conv2d(length,12,kernel_size=(1, dilation_channels),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        x,_,_ = self.block1(x,adj)
        x,d_adj,t_adj = self.block2(x,adj)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x)#b,12,n,1
        return x,d_adj,t_adj
    

class STGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(STGCN,self).__init__()
        tem_size=length
        self.block1=ST_BLOCK_4(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
    def forward(self,input):
        x=self.bn(input)
        adj=self.supports[0]
        
        x=self.block1(x,adj)
        x=self.block2(x,adj)
        x=self.block3(x,adj)
        x=self.conv1(x)#b,12,n,1
        return x,adj,adj 


class OGCRNN(nn.Module):      
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OGCRNN,self).__init__()
       
        self.block1=ST_BLOCK_5(in_dim,dilation_channels,num_nodes,length,K,Kt)
        
        self.tem_size=length
        
        self.conv1=Conv2d(dilation_channels,out_dim,kernel_size=(1,length),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=self.bn(input)
        A=self.h+self.supports[0]
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A=F.dropout(A,0.5)
        x=self.block1(x,A)
        
        x=self.conv1(x)
        return x,A,A   

#OTSGGCN    
class OTSGGCN(nn.Module):
    def __init__(self,device, num_nodes, dropout=0.3, supports=None,length=12, 
                 in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,
                 skip_channels=256,end_channels=512,kernel_size=2,K=3,Kt=3):
        super(OTSGGCN,self).__init__()
        tem_size=length
        self.num_nodes=num_nodes
        self.block1=ST_BLOCK_6(in_dim,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_6(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_6(dilation_channels,dilation_channels,num_nodes,tem_size,K,Kt)
        
        self.conv1=Conv2d(dilation_channels,12,kernel_size=(1, tem_size),padding=(0,0),
                          stride=(1,1), bias=True)
        self.supports=supports
        self.bn=BatchNorm2d(in_dim,affine=False)
        self.h=Parameter(torch.ones(num_nodes,num_nodes), requires_grad=True)
        #nn.init.uniform_(self.h, a=0, b=0.0001)
    def forward(self,input):
        x=input#self.bn(input)
        mask=(self.supports[0]!=0).float()
        A=self.h*mask
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=torch.eye(self.num_nodes).cuda()-A
       # A1=F.dropout(A1,0.5)
        x=self.block1(x,A1)
        x=self.block2(x,A1)
        x=self.block3(x,A1)
        x=self.conv1(x)#b,12,n,1
        return x,A1,A1

    # gwnet


class Graph_WaveNet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, length=12,
                 in_dim=1, out_dim=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(Graph_WaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if supports is None:
            self.supports = []
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope

                additional_scope *= 2

                self.gconv.append(
                    multi_gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_1 = BatchNorm2d(in_dim, affine=False)

    def forward(self, input):
        input = self.bn_1(input)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, adp, adp


class H_GCN(nn.Module):
    def __init__(self, device, num_nodes, cluster_nodes, dropout=0.3, supports=None, supports_cluster=None,
                 transmit=None, length=12,
                 in_dim=1, in_dim_cluster=3, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3, idx=None, idx_c=None):
        super(H_GCN, self).__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.transmit = transmit
        self.cluster_nodes = cluster_nodes
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster,
                                            out_channels=residual_channels,
                                            kernel_size=(1, 1))
        self.supports = supports
        self.supports_cluster = supports_cluster

        self.supports_len = 0
        self.supports_len_cluster = 0
        if supports is not None:
            self.supports_len += len(supports)
            self.supports_len_cluster += len(supports_cluster)

        if supports is None:
            self.supports = []
            self.supports_cluster = []
        self.h = Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        self.h_cluster = Parameter(torch.zeros(cluster_nodes, cluster_nodes), requires_grad=True)
        nn.init.uniform_(self.h_cluster, a=0, b=0.0001)
        self.supports_len += 1
        self.supports_len_cluster += 1
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec1_c = nn.Parameter(torch.randn(cluster_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_c = nn.Parameter(torch.randn(10, cluster_nodes).to(device), requires_grad=True).to(device)

        self.block1 = GCNPool(2 * dilation_channels, dilation_channels, num_nodes, length - 6, 3, dropout, num_nodes,
                              self.supports_len)
        self.block2 = GCNPool(2 * dilation_channels, dilation_channels, num_nodes, length - 9, 2, dropout, num_nodes,
                              self.supports_len)

        self.block_cluster1 = GCNPool(dilation_channels, dilation_channels, cluster_nodes, length - 6, 3, dropout,
                                      cluster_nodes,
                                      self.supports_len)
        self.block_cluster2 = GCNPool(dilation_channels, dilation_channels, cluster_nodes, length - 9, 2, dropout,
                                      cluster_nodes,
                                      self.supports_len)

        self.skip_conv1 = Conv2d(2 * dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(2 * dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.bn = BatchNorm2d(in_dim, affine=False)
        self.conv_cluster1 = Conv2d(dilation_channels, out_dim, kernel_size=(1, 3),
                                    stride=(1, 1), bias=True)
        self.bn_cluster = BatchNorm2d(in_dim_cluster, affine=False)
        self.gate1 = gate(2 * dilation_channels)
        self.gate2 = gate(2 * dilation_channels)
        self.gate3 = gate(2 * dilation_channels)

        self.transmit1 = Transmit(dilation_channels, length, transmit, num_nodes, cluster_nodes)
        self.transmit2 = Transmit(dilation_channels, length - 6, transmit, num_nodes, cluster_nodes)
        self.transmit3 = Transmit(dilation_channels, length - 9, transmit, num_nodes, cluster_nodes)

        self.gc = graph_constructor(num_nodes, 20, 40, device, alpha=3, static_feat=None)
        self.gc_c = graph_constructor(cluster_nodes, 20, 40, device, alpha=3, static_feat=None)

    def forward(self, input, input_cluster):
        x = self.bn(input)
        shape = x.shape
        input_c = input_cluster
        x_cluster = self.bn_cluster(input_c)

        # adp = self.gc(idx)
        # adp_c = self.gc(idx_c)

        if self.supports is not None:
            # nodes
            A = F.relu(torch.mm(self.nodevec1, self.nodevec2))
            d = 1 / (torch.sum(A, -1))
            D = torch.diag_embed(d)
            A = torch.matmul(D, A)

            new_supports = self.supports + [A]
            # region
            A_cluster = F.relu(torch.mm(self.nodevec1_c, self.nodevec2_c))
            d_c = 1 / (torch.sum(A_cluster, -1))
            D_c = torch.diag_embed(d_c)
            A_cluster = torch.matmul(D_c, A_cluster)

            new_supports_cluster = self.supports_cluster + [A_cluster]

        # network
        transmit = self.transmit
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)
        transmit1 = self.transmit1(x, x_cluster)
        x_1 = (torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster))

        x = self.gate1(x, x_1)

        skip = 0
        skip_c = 0
        # 1
        x_cluster = self.block_cluster1(x_cluster, new_supports_cluster)
        x = self.block1(x, new_supports)
        transmit2 = self.transmit2(x, x_cluster)
        x_2 = (torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster))

        x = self.gate2(x, x_2)

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # 2
        x_cluster = self.block_cluster2(x_cluster, new_supports_cluster)
        x = self.block2(x, new_supports)
        transmit3 = self.transmit3(x, x_cluster)
        x_3 = (torch.einsum('bmn,bcnl->bcml', transmit3, x_cluster))

        x = self.gate3(x, x_3)

        s2 = self.skip_conv2(x)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, transmit3, A

class HSTGODE(nn.Module):

    def __init__(self, device, num_nodes, cluster_nodes, dropout=0.3, transmit=None, length=12,
                 in_dim=1, in_dim_cluster=2, out_dim=12, residual_channels=32,skip_channels=256,
                 buildA_true=True, predefined_A=None, predefined_A_c=None,static_feat=None, subgraph_size=20,
                 node_dim=40, dilation_exponential=1, conv_channels=64, end_channels=128, tanhalpha=3, method_1='euler',
                 time_1=1.0, step_size_1=0.25, method_2='euler', time_2=1.0, step_size_2=0.25, alpha=2.0, rtol=1e-4, atol=1e-3,
                 adjoint=False, perturb=False,ln_affine=True):

        super(HSTGODE, self).__init__()

        if method_1 == 'euler':
            self.integration_time = time_1
            self.estimated_nfe = round(self.integration_time / step_size_1)
        elif method_1 == 'rk4':
            self.integration_time = time_1
            self.estimated_nfe = round(self.integration_time / (step_size_1 / 4.0))
        else:
            raise ValueError("Oops! Temporal ODE solver is invaild.")

        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.predefined_A_c = predefined_A_c
        self.seq_length = length
        self.ln_affine = ln_affine
        self.adjoint = adjoint
        self.transmit = transmit
        self.cluster_nodes = cluster_nodes
        self.bn = BatchNorm2d(in_dim, affine=False)

        self.bn_cluster = BatchNorm2d(in_dim_cluster, affine=False)
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.start_conv_cluster = nn.Conv2d(in_channels=in_dim_cluster, out_channels=residual_channels, kernel_size=(1, 1))

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)
        self.gc_c = graph_constructor(cluster_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)
        self.idx = torch.arange(self.num_nodes).to(device)
        self.idx_c = torch.arange(self.cluster_nodes).to(device)


        max_kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (max_kernel_size - 1) * (dilation_exponential ** self.estimated_nfe - 1) / (
                        dilation_exponential - 1))
        else:
            self.receptive_field = self.estimated_nfe * (max_kernel_size - 1) + 1

        if ln_affine:
            self.affine_weight = nn.Parameter(torch.Tensor(*(conv_channels, self.num_nodes)))  # C*H
            self.affine_bias = nn.Parameter(torch.Tensor(*(conv_channels, self.num_nodes)))  # C*H

        self.ODE = ODEBlock(ODEFunc(STBlock(receptive_field=self.receptive_field, dilation=dilation_exponential,
                                            hidden_channels=conv_channels, dropout=self.dropout, method=method_2,
                                            time=time_2, step_size=step_size_2, alpha=alpha, rtol=rtol, atol=atol,
                                            adjoint=False, perturb=perturb)),
                            method_1, step_size_1, rtol, atol, adjoint, perturb)

        self.ODE_C = ODEBlock(ODEFunc(STBlock(receptive_field=self.receptive_field, dilation=dilation_exponential,
                                            hidden_channels=conv_channels, dropout=self.dropout, method=method_2,
                                            time=time_2, step_size=step_size_2, alpha=alpha, rtol=rtol, atol=atol,
                                            adjoint=False, perturb=perturb)),
                            method_1, step_size_1, rtol, atol, adjoint, perturb)


        self.skip_conv1 = Conv2d(2 * residual_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = Conv2d(2 * conv_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.conv_cluster1 = Conv2d(residual_channels, conv_channels, kernel_size=(1, 1),
                                    stride=(1, 1), bias=True)

        self.gate1 = gate(1 * conv_channels)
        self.gate2 = gate(1 * conv_channels)


        self.transmit1 = Transmit(residual_channels, self.receptive_field, transmit, num_nodes, cluster_nodes)
        # self.static_transmit = Static_Transmit(residual_channels,self.receptive_field, transmit,num_nodes,cluster_nodes)
        self.transmit2 = Transmit(conv_channels, length-9 , transmit, num_nodes, cluster_nodes)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=skip_channels*2,
                                  kernel_size=(1,3),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=skip_channels*2,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)
        if ln_affine:
            self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.affine_weight)
        init.zeros_(self.affine_bias)


    def forward(self, input, input_cluster,idx=None, idx_c=None):

        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length, 0))
            input_cluster = nn.functional.pad(input_cluster, (self.receptive_field-self.seq_length, 0))
        if self.buildA_true:
            if idx is None:
                adp = self.gc(self.idx)
                adp_c = self.gc(self.idx_c)
            else:
                adp = self.gc(idx)
                adp_c = self.gc(idx_c)
        else:
            adp = self.predefined_A
            adp_c = self.predefined_A_c



        skip = 0
        skip_c = 0

        x = self.bn(input)
        shape = x.shape
        input_c = input_cluster
        x_cluster = self.bn_cluster(input_c)

        # network
        transmit = self.transmit
        x = self.start_conv(x)
        x_cluster = self.start_conv_cluster(x_cluster)


        transmit1 = self.transmit1(x, x_cluster)
        x_1 = (torch.einsum('bmn,bcnl->bcml', transmit1, x_cluster))
        x = self.gate1(x, x_1)  # 16*64*792*12

        s1 = self.skip_conv1(x)
        skip = s1 + skip

        # 1
        if self.adjoint:
            self.ODE.odefunc.stnet.setIntermediate(dilation=1)
            self.ODE_C.odefunc.stnet.setIntermediate(dilation=1)
        self.ODE.odefunc.stnet.setGraph(adp)
        self.ODE_C.odefunc.stnet.setGraph(adp_c)
        x = self.ODE(x, self.integration_time)
        x_cluster = self.conv_cluster1(x_cluster)
        x_cluster = self.ODE_C(x_cluster, self.integration_time)
        self.ODE.odefunc.stnet.setIntermediate(dilation=1)
        self.ODE_C.odefunc.stnet.setIntermediate(dilation=1)

        x = x[..., -3:]
        x_cluster = x_cluster[..., -3:]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)
        x_cluster = F.layer_norm(x_cluster, tuple(x_cluster.shape[1:]), weight=None, bias=None, eps=1e-5)

        transmit2 = self.transmit2(x, x_cluster)
        x_2 = (torch.einsum('bmn,bcnl->bcml', transmit2, x_cluster))
        x = self.gate2(x, x_2)  # 16*64*792*6


        s2=self.skip_conv2(x)
        skip = skip[:, :, :,  -s2.size(3):]
        skip = s2 + skip

        #output
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x, transmit2




