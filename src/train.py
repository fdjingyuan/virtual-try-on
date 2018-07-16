#coding: utf-8
from src.dataset import DeepFashionInShopDataset
import torch
import torch.utils.data
from torch import nn
from torch.optim import lr_scheduler
import numpy as np
from torch.nn import functional as F
from src import const
from src.center_loss import CenterLoss
from src.utils import get_train_test
from tensorboardX import SummaryWriter
import os
import argparse


parser = argparse.ArgumentParser("Center Loss Example")

# optimization
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_model', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--lr_cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--epoch', type=int, default=50)
# model
#parser.add_argument('--model', type=str, default='cnn')
#misc
parser.add_argument('--print-freq', type=int, default=20)

args = parser.parse_args()


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
    print("Batch_size:{}; Model learning rate:{}; Center learning rate:{}; Total Epoch:{}; "
            .format(args.batch_size,args.lr_model,args.lr_cent,args.epoch))
    net = const.USE_NET(const.NUM_CLASSES)
    net = net.to(const.device)  # 转移到cpu/gpu上
  
    #center loss and parameters
    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=const.NUM_CLASSES, feat_dim=const.FEATURE_EMBEDDING, use_gpu=True)
  
    optimizer_model = torch.optim.SGD(net.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)


    #center_loss = CenterLoss(const.NUM_CLASSES,2048,True)
    #params = list(net.parameters()) + list(center_loss.parameters())
    #optimizer = torch.optim.SGD(params, lr=learning_rate)

    #write to tensorboardX
    writer = SummaryWriter(const.TRAIN_DIR)

    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=const.STEP_SIZE, gamma=const.LEARNING_RATE_DECAY)
    step = 0
    for epoch in range(args.epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.epoch))
        train(net,criterion_xent, criterion_cent, optimizer_model, optimizer_centloss,
              train_dataloader, const.NUM_CLASSES, epoch, writer, step)
        test(net, test_dataloader, const.NUM_CLASSES, epoch, writer, step)
        
        step += 1
      
        print('Saving Model....')
        torch.save(net.state_dict(), 'models/' + const.MODEL_NAME ) 
    print('Finished')


def train(net, criterion_xent, criterion_cent, optimizer_model, optimizer_centloss,
          trainloader, num_classes, epoch, writer, step):       
    net.train()

    for i, sample in enumerate(trainloader):
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

        if (i + 1) % args.print_freq == 0:
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('learning_rate', args.lr_model, step)
            print("Batch {}/{}\t Loss {:.6f} \t XentLoss {:.6f} \t CenterLoss {:.6f}" \
                  .format(i+1, len(trainloader), loss.item(), loss_xent.item(), loss_cent.item()))
            

        
def test(net, testloader, num_classes, epoch, writer, step):
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    correct, total = 0, 0
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            for key in sample:
                sample[key] = sample[key].to(const.device) #transfer to cuda(gpu)
            output = net(sample['image'])['output']
            feature = net(sample['image'])['embedding']
            _, predicted = torch.max(output.data, 1)
            total += sample['label'].size(0)
            correct += (predicted == sample['label']).sum().item()

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        writer.add_scalar('accuracy', correct / total, step)


if __name__ == '__main__':   
    main()
