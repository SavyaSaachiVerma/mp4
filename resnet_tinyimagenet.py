import os

import numpy as np
import torch
import torch.nn as nn
import random
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.backends import cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
from torchvision import datasets

from resnetmodel import ResNet
from basicblock import BasicBlock

def create_val_folder(val_dir):
    """
    This method is responsible for separating validation
    images into separate sub folders
    """
    # path where validation data is present now
    path = os.path.join(val_dir, 'images')
    # file where image2class mapping is present
    filename = os.path.join(val_dir, 'val_annotations.txt')
    fp = open(filename, "r") # open file in read mode
    data = fp.readlines() # read line by line
    """
    Create a dictionary with image names as key and
    corresponding classes as values
    """
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
    val_img_dict[words[0]] = words[1]
    fp.close()
    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
    if not os.path.exists(newpath):  # check if folder exists
        os.makedirs(newpath)
    # Check if image exists in default directory
    if os.path.exists(os.path.join(path, img)):
        os.rename(os.path.join(path, img), os.path.join(newpath, img))
    return

batch_size_train = 128
batch_size_test = 64
learning_rate = 0.001
epochs = 30
load_chkpt = False

def main():

    """Transformations for Augmenting and Normalizing Training Dataset"""
    augment_train_ds = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])
    """Normalizing Test Dataset"""
    augment_test_ds = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    """Set seed."""
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_dir = '/u/training/tra287/scratch/tiny-imagenet-200/train'
    train_ds = datasets.ImageFolder(train_dir,transform=augment_train_ds)
    train_ds_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=8)
    val_dir = '/u/training/tra287/scratch/tiny-imagenet-200/val'
    val_img_dir = os.path.join(val_dir, 'images')
    # if 'val_' in os.listdir(val_img_dir)[0]:
    #     create_val_folder(val_dir)
    # else:
    #     pass
    test_ds = datasets.ImageFolder(val_dir, transform=augment_test_ds)
    test_ds_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size_test, shuffle=False, num_workers=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Initializing Model")
    basic_block = BasicBlock
    res_net = ResNet(basic_block=basic_block, num_basic_blocks_list=[2, 4, 4, 2], num_classes=100)
    res_net = res_net.to(device)
    start_epoch = 0

    if load_chkpt:
        print("Saved Model is being loaded")
        chkpt = torch.load('./Checkpoint/model_state.pt')
        res_net.load_state_dict(chkpt['res_net_model'])
        start_epoch = chkpt['epoch']

    """If multiple GPUs are available then use asynchronous training """
    if device == 'cuda':
        res_net = torch.nn.DataParallel(res_net)
        cudnn.benchmark = True

    """___________ Training ___________"""

    print("Starting Training")

    """Criterion Function: Softmax + Log-Likelihood"""
    loss_fn = nn.CrossEntropyLoss()
    """Adam Optimizer (as it takes advantage of both RMSDrop and Momentum"""
    optimizer = optim.Adam(res_net.parameters(), lr=learning_rate)

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
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            """Loss function requires the inputs to be wrapped in variables"""
            # inputs = Variable(inputs)

            """Torch tends to take cumulative gradients which is not required so setting it to zero after each batch"""
            optimizer.zero_grad()

            outputs = res_net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            cur_loss += loss.item()
            cur_loss /= (i + 1)

            _, predicted_label = torch.max(outputs, 1)
            # print(predicted_label.shape, labels.shape)
            total_samples += labels.shape[0]
            # arr = (predicted_label == labels).numpy()
            # print(np.sum(arr))
            """can not use numpy as the tensors are in CUDA"""
            total_correct += predicted_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            if i % 20 == 0:
                print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                      (epoch + 1, i + 1, cur_loss, accuracy))

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

    """___________ Testing ____________"""
    print("Testing Started")
    """Puts model in testing state"""
    res_net.eval()

    cur_loss = 0.0
    total_correct = 0
    total_samples = 0
    """Do testing under the no_grad() context so that torch does not store/use these actions to calculate gradients"""
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_ds_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = Variable(inputs)

            outputs = res_net(inputs)
            loss = loss_fn(outputs, labels)

            cur_loss += loss.item()
            cur_loss /= (i + 1)

            _, predicted_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            # arr = (predicted_label == labels).numpy()
            total_correct += predicted_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            if i % 50 == 0:
                print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
                      (i + 1, cur_loss, accuracy))

    print("Testing Completed with accuracy:" + str(accuracy))


if __name__ == "__main__":
    main()



