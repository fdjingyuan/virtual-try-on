#coding: utf-8
from src.dataset import DeepFashionInShopDataset
import torch
import torch.utils.data
from torch import nn
import numpy as np
from torch.nn import functional as F
from src import utils
from src import const
from scr.center_loss import CenterLoss
from src.utils import get_train_test
from tensorboardX import SummaryWriter
import os



def main():
    if os.path.exists('models') is False:
        os.makedirs('models')

    # get train, test dataloader
    train_df, test_df = get_train_test()
    train_dataset = DeepFashionInShopDataset(train_df, 'RANDOM')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataset = DeepFashionInShopDataset(test_df, 'CENTER')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    #get network
    print("Creating model: {}".format(const.net_name))
    net = const.USE_NET(const.NUM_CLASSES)
    net = net.to(const.device)  # 转移到cpu/gpu上

    #set learning rate and optimizer
    learning_rate = const.LEARNING_RATE

    #center loss and parameters
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=const.NUM_CLASSES, feat_dim=2048, use_gpu=True)
    optimizer_model = torch.optim.SGD(net.parameters(), lr=learning_rate) 
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=learning_rate)

    #center_loss = CenterLoss(const.NUM_CLASSES,2048,True)
    #params = list(net.parameters()) + list(center_loss.parameters())
    #optimizer = torch.optim.SGD(params, lr=learning_rate)

    #write to tensorboardX
    writer = SummaryWriter(const.TRAIN_DIR)

    total_step = len(train_dataloader)
    step = 0
    criterion = center_loss

    for epoch in range(const.NUM_EPOCH):
        train(net,criterion_xent, criterion_cent,
              optimizer_model, optimizer_centloss,
              train_dataloader, const.NUM_CLASSES, epoch)
        
        

        print('Saving Model....')
        torch.save(net.state_dict(), 'models/' + const.MODEL_NAME)
        print('OK. Now evaluate..')




def train(net, criterion_xent, criterion_cent, optimizer_model, optimizer_centloss,
          trainloader, num_classes, epoch):       
    net.train()

    for i, sample in enumerate(trainloader):
        step += 1
        for key in sample:
            sample[key] = sample[key].to(const.device)

        output = net(sample['image'])
        loss_xent = criterion_xent(output['output'], sample['label'])
        loss_cent = criterion_cent(output['embedding'], sample['label'])
        loss_cent *= const.WEIGHT_CENT
        loss = loss_xent + loss_cent

        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / const.WEIGHT_CENT)
        optimizer_centloss.step()

        if (i + 1) % 10 == 0:
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('learning_rate', learning_rate, step)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, const.NUM_EPOCH, i + 1, total_step, loss.item()))

        
def test(model, testloader, num_classes, epoch)
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, sample in enumerate(test_dataloader):
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample['image'])['output']
            _, predicted = torch.max(output.data, 1)
            total += sample['label'].size(0)
            correct += (predicted == sample['label']).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        writer.add_scalar('accuracy', correct / total, step)
    # learning rate decay
    learning_rate *= const.LEARNING_RATE_DECAY
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


if __name__ == '__main__':
    main()