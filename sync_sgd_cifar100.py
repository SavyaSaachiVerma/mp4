import os

import numpy as np
import torch
import torch.nn as nn
import random
import torchvision
import torchvision.transforms as transforms
import csv

from torch import optim
from torch.backends import cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable

from data_partition_helper import DataPartitioner
from resnetmodel import ResNet
from basicblock import BasicBlock

import torch.distributed as dist
import subprocess
from mpi4py import MPI

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]
name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())
ip = comm.gather(ip)
if rank != 0:
    ip = None
ip = comm.bcast(ip, root=0)
os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'
backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)
dtype = torch.FloatTensor


batch_size_train = 128
batch_size_test = 64
data_dir = "./Data"
learning_rate = 0.001
epochs = 5
load_chkpt = False

""" Partitioning MNIST """
def partition_dataset():

    """Transformations for Augmenting and Normalizing Training Dataset"""
    augment_train_ds = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])
    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                             transform=augment_train_ds)
    size = dist.get_world_size()
    bsz = batch_size_train / size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(train_ds, partition_sizes)
    partition = partition.use(rank)
    train_set = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        tensor0 = param.grad.data.cpu()
        dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
        tensor0 /= size
        param.grad.data = tensor0.cuda()

def main():

    train_ds = None
    if rank == 0:
        """Normalizing Test Dataset"""
        augment_test_ds = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
        ])
        """Loading the datasets from torchvision dataset library"""
        test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                transform=augment_test_ds)
        test_ds_loader = data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False, num_workers=0)

    """Set seed"""
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Initializing Model")
    basic_block = BasicBlock
    res_net = ResNet(basic_block=basic_block, num_basic_blocks_list=[2, 4, 4, 2], num_classes=100)
    res_net = res_net.cuda()
    start_epoch = 0

    train_ds_loader, batch_size_train = partition_dataset()

    if rank == 0:
        if load_chkpt:
            print("Saved Model is being loaded")
            chkpt = torch.load('./Checkpoint/model_state.pt')
            res_net.load_state_dict(chkpt['res_net_model'])
            start_epoch = chkpt['epoch']

    """___________ Training ___________"""

    print("Starting Training")

    """Criterion Function: Softmax + Log-Likelihood"""
    loss_fn = nn.CrossEntropyLoss()
    """Adam Optimizer (as it takes advantage of both RMSDrop and Momentum"""
    optimizer = optim.Adam(res_net.parameters(), lr=learning_rate)

    test_acc_list = []

    for epoch in range(start_epoch, epochs):

        cur_loss = 0.0
        total_correct = 0
        total_samples = 0

        """ Overflow error in the optimizer if the step size is not reset."""
        if epoch > 8:
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if state['step'] >= 1024:
                        state['step'] = 1000

        for i, (inputs, labels) in enumerate(train_ds_loader):

            """Transfer inputs and labels to CUDA if available"""
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            """Loss function requires the inputs to be wrapped in variables"""
            # inputs = Variable(inputs)

            """Torch tends to take cumulative gradients which is not required so setting it to zero after each batch"""
            optimizer.zero_grad()

            outputs = res_net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            average_gradients(res_net)
            optimizer.step()

            cur_loss += loss.cpu().data.numpy()
            avg_loss = cur_loss / (i + 1)

            _, predicted_label = torch.max(outputs, 1)
            # print(predicted_label.shape, labels.shape)
            total_samples += labels.shape[0]
            # arr = (predicted_label == labels).numpy()
            # print(np.sum(arr))
            """can not use numpy as the tensors are in CUDA"""
            total_correct += predicted_label.eq(labels.long()).float().sum().cpu().data.numpy()
            accuracy = total_correct / total_samples

            if i % 100 == 0:
                print('Training [rank: %d, epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                      (rank, epoch + 1, i + 1, avg_loss, accuracy))

        if rank == 0:

            test_acc_list.append(test(loss_fn, res_net, test_ds_loader))

            """Saving model after every 5 epochs"""
            if (epoch + 1) % 5 == 0:
                print('==> Saving model ...')
                state = {
                    'res_net_model': res_net.state_dict(),
                    'epoch': epoch,
                }
                if not os.path.isdir('./Checkpoint'):
                    os.mkdir('Checkpoint')
                torch.save(state, './Checkpoint/model_state.pt')

    print("Training Completed!")

    if rank == 0:
        """___________ Testing ____________"""
        print("Testing Started")
        """Puts model in testing state"""
        res_net.eval()

        accuracy = test(loss_fn, res_net, test_ds_loader)

        print("Testing Completed with accuracy:" + str(accuracy))

        # save the test accuracy list in csv file
        with open('graph_sync_sgd_cifar100.csv', 'wb') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerow(test_acc_list)

        print("Saved Test Accuracy list for graph")


def test(loss_fn, res_net, test_ds_loader):
    cur_loss = 0.0
    total_correct = 0
    total_samples = 0
    """Do testing under the no_grad() context so that torch does not store/use these actions to calculate gradients"""
    # with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_ds_loader):
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        # inputs = Variable(inputs)

        outputs = res_net(inputs)
        loss = loss_fn(outputs, labels)

        cur_loss += loss.cpu().data.numpy()
        avg_loss = cur_loss / (i + 1)

        _, predicted_label = torch.max(outputs, 1)
        total_samples += labels.shape[0]
        total_correct += predicted_label.eq(labels.long()).float().sum().cpu().data.numpy()
        accuracy = total_correct / total_samples

        if i % 50 == 0:
            print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' % (i + 1, avg_loss, accuracy))
    return accuracy


if __name__=="__main__":
    main()
