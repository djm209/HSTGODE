import torch.optim as optim
from model import *
import util

class trainer1():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = ASTGCN(device, num_nodes, dropout, supports=supports,
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse    


class trainer2():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = STGCN(device, num_nodes, dropout, supports=supports,
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse     


class trainer3():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = OGCRNN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse    
    
class trainer4():
    def __init__(self, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports,decay):
        self.model = OTSGGCN(device, num_nodes, dropout, supports=supports, 
                           in_dim=in_dim, out_dim=seq_length, 
                           residual_channels=nhid, dilation_channels=nhid, 
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae
        
        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output

        loss = self.loss(predict, real,0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        #input = nn.functional.pad(input,(1,0,0,0))
        output,_,_ = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        
        predict = output
        loss = self.loss(predict, real,0.0)
        mae = util.masked_mae(predict,real,0.0).item()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mae,mape,rmse   
    
# hgcn
class trainer5():
    def __init__(self, in_dim, in_dim_cluster, seq_length, num_nodes, cluster_nodes, nhid, dropout, lrate, wdecay,
                 device, supports, supports_cluster, transmit, decay):
        self.model = H_GCN(device, num_nodes, cluster_nodes, dropout, supports=supports,
                           supports_cluster=supports_cluster,
                           in_dim=in_dim, in_dim_cluster=in_dim_cluster, out_dim=seq_length, transmit=transmit,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16, idx=None, idx_c=None)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5
        self.supports = supports
        self.num_nodes = num_nodes

    def train(self, input, input_cluster, real_val, real_val_cluster):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, output_cluster, tran2 = self.model(input, input_cluster)
        output = output.transpose(1, 3)
        # output_cluster = output_cluster.transpose(1,3)
        # output = [batch_size,1,num_nodes,12]
        real = torch.unsqueeze(real_val, dim=1)
        # real_cluster = real_val_cluster[:,1,:,:]
        # real_cluster = torch.unsqueeze(real_cluster,dim=1)
        predict = output

        loss = self.loss(predict, real, 0.0)  # +energy
        # loss1 =self.loss(output_cluster, real_cluster,0.0)
        # print(loss)
        (loss).backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, input_cluster, real_val, real_val_cluster):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input, input_cluster)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse




class trainer6():
    def __init__(self, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, decay):
        self.model = Graph_WaveNet(device, num_nodes, dropout, supports=supports,
                           in_dim=in_dim, out_dim=seq_length,
                           residual_channels=nhid, dilation_channels=nhid,
                           skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss = util.masked_mae

        self.clip = 5

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        # input = nn.functional.pad(input,(1,0,0,0))
        output, _, _ = self.model(input)
        output = output.transpose(1, 3)
        # output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val, dim=1)

        predict = output
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

class trainer8():
    def __init__(self, args,transmit, predefined_A, predefined_A_c, scaler, device, cl=True):
        self.scaler = scaler
        self.model = HSTGODE(device, num_nodes=args.num_nodes, cluster_nodes=args.cluster_nodes, dropout=args.dropout, transmit=transmit, length=args.seq_in_len,
                   in_dim=args.in_dim, in_dim_cluster=args.in_dim_cluster, out_dim=args.seq_out_len, residual_channels=args.nhid,skip_channels=args.skip_channels,
                   buildA_true=args.buildA_true, predefined_A=predefined_A, predefined_A_c=predefined_A_c, subgraph_size=args.subgraph_size,
                   node_dim=args.node_dim, dilation_exponential=args.dilation_exponential, conv_channels=args.conv_channels, end_channels=args.end_channels, tanhalpha=args.tanhalpha,
                   method_1=args.solver_1, time_1=args.time_1,
                   step_size_1=args.step_1, method_2=args.solver_2, time_2=args.time_2, step_size_2=args.step_2,
                   alpha=args.alpha, rtol=args.rtol, atol=args.atol, adjoint=args.adjoint, perturb=args.perturb,
                   ln_affine=True)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.loss = util.masked_mae
        # self.loss = util.masked_mse
        # self.loss = util.masked_rmses
        self.clip = args.clip
        self.step = args.step_size1
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = args.seq_out_len
        self.cl = cl

    def train(self, input,input_cluster,real_val,real_val_cluster, idx=None, idx_c=None):
        self.model.train()
        self.optimizer.zero_grad()
        output, _ = self.model(input, input_cluster, idx=idx, idx_c=idx_c)
        output = output.transpose(1, 3)

        self.model.ODE.odefunc.nfe = 0  # reset CTA nfe
        self.model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
        self.model.ODE.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe

        self.model.ODE_C.odefunc.nfe = 0  # reset CTA nfe
        self.model.ODE_C.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
        self.model.ODE_C.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe


        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        if self.iter % self.step == 0 and self.task_level <= self.seq_out_len:
            self.task_level += 1
        if self.cl:
            loss = self.loss(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = self.loss(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        mae = util.masked_mae(predict, real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()
        self.iter += 1

        return loss.item(), mae, mape, rmse

    def eval(self, input, input_cluster, real_val, real_val_cluster):
        self.model.eval()
        output, _ = self.model(input, input_cluster)
        self.model.ODE.odefunc.nfe = 0  # reset CTA nfe
        self.model.ODE.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
        self.model.ODE.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe
        self.model.ODE_C.odefunc.nfe = 0  # reset CTA nfe
        self.model.ODE_C.odefunc.stnet.gconv_1.CGPODE.odefunc.nfe = 0  # reset CGP 1 nfe
        self.model.ODE_C.odefunc.stnet.gconv_2.CGPODE.odefunc.nfe = 0  # reset CGP 2 nfe

        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mae = util.masked_mae(predict,real, 0.0).item()
        mape = util.masked_mape(predict, real, 0.0).item()
        rmse = util.masked_rmse(predict, real, 0.0).item()

        return loss.item(), mae, mape, rmse


