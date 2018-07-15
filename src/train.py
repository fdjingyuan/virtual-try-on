#coding: utf-8
from src.dataset import DeepFashionInShopDataset
import torch
import torch.utils.data
from torch import nn
import numpy as np
from torch.nn import functional as F
from src import const
from src.utils import parse_args_and_merge_const, get_train_test
from tensorboardX import SummaryWriter
import os


if __name__ == '__main__':
    parse_args_and_merge_const()
    if os.path.exists('models') is False:
        os.makedirs('models')

    # get train, test dataloader
    train_df, test_df = get_train_test()
    train_dataset = DeepFashionInShopDataset(train_df, 'RANDOM')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataset = DeepFashionInShopDataset(test_df, 'CENTER')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=const.VAL_BATCH_SIZE, shuffle=True, num_workers=4)

    #get network
    net = const.USE_NET(const.NUM_CLASSES)
    net = net.to(const.device)  # 转移到cpu/gpu上

    #set learning rate and optimizer
    learning_rate = const.LEARNING_RATE
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    #write to tensorboardX
    writer = SummaryWriter(const.TRAIN_DIR)

    total_step = len(train_dataloader)
    step = 0
    criterion = nn.CrossEntropyLoss()
    for epoch in range(const.NUM_EPOCH):
        net.train()
        for i, sample in enumerate(train_dataloader):
            step += 1
            for key in sample:
                sample[key] = sample[key].to(const.device)
            output = net(sample['image'])
            loss = criterion(output['output'], sample['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('learning_rate', learning_rate, step)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, const.NUM_EPOCH, i + 1, total_step, loss.item()))

        print('Saving Model....')
        torch.save(net.state_dict(), 'models/' + const.MODEL_NAME)
        print('OK. Now evaluate..')

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
