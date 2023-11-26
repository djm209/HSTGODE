import argparse
import shutil
import time
import numpy as np

import util
from engine import trainer8
from util import *
import torch.optim as optim


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description='MTGODE')


parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/XiAN_City/cluster_40',help='data path')
parser.add_argument('--filename',type=str,default='XiAN_City',help='filename')
parser.add_argument('--adjdata',type=str,default='data/XiAN_City/cluster_40/adj_mat.pkl',help='adj data path')
parser.add_argument('--adjdatacluster',type=str,default='data/XiAN_City/cluster_40/adj_mat_cluster.pkl',help='adj data path')
parser.add_argument('--transmit',type=str,default='data/XiAN_City/cluster_40/transmit.csv',help='data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--nhid',type=int,default=32,help='')

# general settings
parser.add_argument('--expid', type=int, default=0, help='experiment id when saving best model')
parser.add_argument('--runs', type=int, default=1, help='number of runs')
parser.add_argument('--buildA_true', type=str_to_bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--save',type=str,default='./garage/XiAN_City/culster_40_test',help='save path')
parser.add_argument('--save_preds', type=str_to_bool, default=True, help='whether to save prediction results')
parser.add_argument('--save_preds_path', type=str, default='./garage/', help='predictions save path')
parser.add_argument('--num_nodes',type=int,default=792,help='number of nodes')
parser.add_argument('--cluster_nodes',type=int,default=40,help='number of cluster')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--in_dim_cluster',type=int,default=2,help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=12, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=12, help='output sequence length')
parser.add_argument("--force", type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--model',type=str,default='HSTGODE',help='adj type')

# training related
parser.add_argument('--print_every', type=int, default=100, help='')
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--lr_decay', type=str_to_bool, default=True, help='whether to decrease lr during training')
parser.add_argument('--lr_decay_steps', type=int, default=100, help='lr decay at this step')
parser.add_argument('--lr_decay_rate', type=float, default=0.92,help='how much lr will decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--clip', type=int, default=5, help='clip')
parser.add_argument('--step_size1', type=int, default=2500, help='control the curriculum learning')
parser.add_argument('--step_size2', type=int, default=100, help='control the node permutation')
parser.add_argument('--cl', type=str_to_bool, default=True, help='whether to do curriculum learning')
# model related
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--subgraph_size', type=int, default=20, help='learned adj top-k sparse')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--tanhalpha', type=float, default=3, help='saturation ratio in graph construction')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=64, help='convolution channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')
parser.add_argument('--skip_channels', type=int, default=256, help='end channels')
parser.add_argument('--solver_1', type=str, default='euler', help='CTA Solver')
parser.add_argument('--time_1', type=float, default=1, help='CTA integration time')
parser.add_argument('--step_1', type=float, default=0.25, help='CTA step size')
parser.add_argument('--solver_2', type=str, default='euler', help='CGP Solver')
parser.add_argument('--time_2', type=float, default=1.0, help='CGP integration time')
parser.add_argument('--step_2', type=float, default=0.25, help='CGP step size')
parser.add_argument('--alpha', type=float, default=2.0, help='eigen normalization')
parser.add_argument('--rtol', type=float, default=1e-4, help='rtol')
parser.add_argument('--atol', type=float, default=1e-3, help='atol')
parser.add_argument('--adjoint', type=str_to_bool, default=False, help='whether to use adjoint method')
parser.add_argument('--perturb', type=str_to_bool, default=False, help='')

args = parser.parse_args()
torch.set_num_threads(4)


def main(runid):
    # load data
    early_stop = 0
    device = torch.device(args.device)
    dataloader = util.load_dataset_cluster(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    transmit_np=np.float32(np.loadtxt(args.transmit,delimiter=','))
    transmit=torch.tensor(transmit_np).to(device)

    # load predefined adj
    predefined_A = load_mtogde_adj(args.adjdata)
    predefined_A = torch.tensor(predefined_A) - torch.eye(args.num_nodes)  # remove self-loop cuz we do it later
    predefined_A = predefined_A.to(device)
    predefined_A_c = load_mtogde_adj(args.adjdatacluster)
    predefined_A_c = torch.tensor(predefined_A_c) - torch.eye(args.cluster_nodes)  # remove self-loop cuz we do it later
    predefined_A_c = predefined_A_c.to(device)

    if args.model == 'HSTGODE':
        engine = trainer8(args, transmit, predefined_A, predefined_A_c, scaler, device, cl=True)

    if args.lr_decay:
        lr_decay_steps = [args.lr_decay_steps]
        scheduler = optim.lr_scheduler.MultiStepLR(engine.optimizer, milestones=lr_decay_steps, gamma=args.lr_decay_rate)

    print(args)

    # check parameters file
    params_path = args.save + "/" + args.model
    if os.path.exists(params_path) and not args.force:
        raise SystemExit("Params folder exists! Select a new params path please!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    """
    Epoch training
    """
    print("\nstart training...", flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5

    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader_cluster'].shuffle()

        for iter,(x,y,x_cluster,y_cluster) in enumerate(dataloader['train_loader_cluster'].get_iterator()):
            # trainx.shape = (batch, in_dim, num_nodes, seq_in_len)
            trainx = torch.FloatTensor(x).transpose(1, 3).to(device)
            trainy = torch.FloatTensor(y).transpose(1, 3).to(device)
            trainx_cluster = torch.FloatTensor(x_cluster).transpose(1, 3).to(device)
            trainy_cluster = torch.FloatTensor(y_cluster).transpose(1, 3).to(device)
            if iter % args.step_size2 == 0:
                perm = np.random.permutation(range(args.num_nodes))
                perm_c = np.random.permutation(range(args.cluster_nodes))

            num_sub = int(args.num_nodes/args.num_split)
            num_sub_c = int(args.cluster_nodes/args.num_split)

            for j in range(args.num_split):
                if j != args.num_split-1:
                    id = perm[j * num_sub:(j + 1) * num_sub]
                    id_c = perm_c[j * num_sub_c:(j + 1) * num_sub_c]
                else:
                    id = perm[j * num_sub:]
                    id_c = perm_c[j * num_sub_c:]
                id = torch.tensor(id).to(device).type(torch.long)
                id_c = torch.tensor(id_c).to(device).type(torch.long)

                tx = trainx[:, :, id, :]
                ty = trainy[:, :, id, :]
                tx_cluster = trainx_cluster[:, :, id_c, :]
                ty_cluster = trainy_cluster[:, :, id_c, :]
                metrics = engine.train(tx, tx_cluster, ty[:, 0, :, :], ty_cluster[:, 0, :, :], id, id_c)
                train_loss.append(metrics[0])
                train_mape.append(metrics[1])
                train_rmse.append(metrics[2])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}  Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

        t2 = time.time()
        train_time.append(t2-t1)

        # validation after each epoch
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()

        for iter,(x,y,x_cluster,y_cluster) in enumerate(dataloader['val_loader_cluster'].get_iterator()):
            testx = torch.FloatTensor(x).transpose(1, 3).to(device)
            testx_cluster = torch.FloatTensor(x_cluster).transpose(1, 3).to(device)
            testy = torch.FloatTensor(y).transpose(1, 3).to(device)
            testy_cluster = torch.FloatTensor(y_cluster).transpose(1, 3).to(device)
            metrics = engine.eval(testx, testx_cluster, testy[:, 0, :, :], testy_cluster[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)

        # save the best model for this run over epochs
        if mvalid_loss < minl:
            torch.save(engine.model.state_dict(), params_path + "/" + args.model + "_exp" + str(args.expid) + "_" + str(runid) +".pth")
            print("BEST MODEL UPDATE")
            minl = mvalid_loss
            early_stop = 0
        else:
            early_stop = early_stop + 1

        if early_stop > 10:
            break

        if args.lr_decay:
            scheduler.step()  # adjust learning rate

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(params_path + "/" + args.model + "_exp" + str(args.expid) + "_" + str(runid) +".pth"))

    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))

    """
    Model evaluation
    """
    # validation on the best model

    outputs = []
    realy = torch.FloatTensor(dataloader['y_val']).transpose(1, 3)[:, 0, :, :].to(device)

    for iter, (x, y, x_cluster, y_cluster) in enumerate(dataloader['val_loader_cluster'].get_iterator()):
        valx = torch.Tensor(x).to(device)
        valx = valx.transpose(1, 3)
        valx_cluster = torch.Tensor(x_cluster).to(device)
        valx_cluster = valx_cluster.transpose(1, 3)
        with torch.no_grad():
            preds, _ = engine.model(valx, valx_cluster)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    pred = scaler.inverse_transform(yhat)
    vmae, vmape, vrmse = metric(pred, realy)

    # testing on the best model
    outputs = []
    realy = torch.FloatTensor(dataloader['y_test']).transpose(1, 3)[:, 0, :, :].to(device)

    for iter, (x, y, x_cluster, y_cluster) in enumerate(dataloader['test_loader_cluster'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        testx_cluster = torch.Tensor(x_cluster).to(device)
        testx_cluster = testx_cluster.transpose(1, 3)
        with torch.no_grad():
            preds, _ = engine.model(testx, testx_cluster)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    mae = []
    mape = []
    rmse = []

    for i in range(args.seq_out_len):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])

    if args.save_preds:
        prediction_path = params_path + "/" + args.model + "_prediction_results"
        ground_truth = realy.detach().cpu().numpy()
        prediction = scaler.inverse_transform(yhat).detach().cpu().numpy()
        np.savez_compressed(
            os.path.normpath(prediction_path),
            prediction=prediction,
            ground_truth=ground_truth
        )

    return vmae, vmape, vrmse, mae, mape, rmse


if __name__ == "__main__":
    vmae = []
    vmape = []
    vrmse = []
    mae = []
    mape = []
    rmse = []

    for i in range(args.runs):
        vm1, vm2, vm3, m1, m2, m3 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)

    amae = np.mean(mae, 0)
    amape = np.mean(mape, 0)
    armse = np.mean(rmse, 0)

    smae = np.std(mae, 0)
    smape = np.std(mape, 0)
    srmse = np.std(rmse, 0)

    print('\n\n==========Results for multiple runs==========\n\n')

    # validation avg and std over multiple runs
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape)))
    print('\n\n')

    # testing avg and std over multiple runs
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], smae[i], srmse[i], smape[i]))