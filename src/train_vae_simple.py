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
from src.perceptual_loss import VGG_perceptual_loss_16
from src.utils import get_train_test
from tensorboardX import SummaryWriter
import os
import argparse
from torchvision.utils import save_image

parser = argparse.ArgumentParser("embedding")

# optimization
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr_model', type=float, default=0.0001, help="learning rate for model")
parser.add_argument('--lr_cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--lr_ploss', type=float, default=0.01, help="learning rate for perceptual loss")
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

    optimizer_model = torch.optim.SGD(net.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)

    #write to tensorboardX
    writer = SummaryWriter(const.TRAIN_DIR)
    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=const.STEP_SIZE, gamma=const.LEARNING_RATE_DECAY)
    step = 0
    for epoch in range(args.epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.epoch))
        train(net, optimizer_model, train_dataloader, const.NUM_CLASSES, epoch, writer, step)
        test(net, test_dataloader, const.NUM_CLASSES, epoch, writer, step)

        step += 1

        print('Saving Model....')
        torch.save(net.state_dict(), 'models/' + const.MODEL_NAME ) 
        print('Finished')

def train(net, optimizer_model, trainloader, num_classes, epoch, writer, step):       
    net.train()

    for i, sample in enumerate(trainloader):
        for key in sample:
            sample[key] = sample[key].to(const.device)
        output = net(sample['image'])
        # loss: KLD
        loss_kld = output['KLD']
        # loss: classification
        loss_clf = F.cross_entropy(output['output'], sample['label'])
        # loss: BCE (copy from pytorch的例子)，但是去掉了size_average=False，不然loss非常大
        # image_tensor: 0~1的图像数据，因为F.binary_cross_entropy要求两者都在0~1之间，参考：https://github.com/Lasagne/Recipes/issues/54
        origin_x = sample['image_tensor'].reshape(sample['image_tensor'].shape[0], -1)
        recon_x = output['decoded'].reshape(output['decoded'].shape[0], -1)
        loss_bce = F.binary_cross_entropy(recon_x, origin_x)
        
        loss = loss_kld * .0001 + loss_clf * 0+ loss_bce

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        if (i + 1) % args.print_freq == 0:
            writer.add_scalar('loss', loss.item(), step)
            writer.add_scalar('loss_kld', loss_kld.item(), step)
            writer.add_scalar('loss_clf', loss_clf.item(), step)
            writer.add_scalar('loss_bce', loss_bce.item(), step)
            writer.add_scalar('learning_rate', args.lr_model, step)
            print("Batch {}/{}\t Loss {:.6f} \t loss_kld {:.6f} \t loss_clf {:.6f} \t loss_bce {:.6f}" \
                  .format(i+1, len(trainloader), loss.item(), loss_kld.item(), loss_clf.item(), loss_bce.item()))

        
def test(net, testloader, num_classes, epoch, writer, step):
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    correct, total = 0, 0
    if not(os.path.exists('results')):
        os.makedirs('results')
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            for key in sample:
                sample[key] = sample[key].to(const.device) #transfer to cuda(gpu)
            output = net(sample['image'])
            feature = net(sample['image'])['embedding']
            _, predicted = torch.max(output['output'].data, 1)
            total += sample['label'].size(0)
            correct += (predicted == sample['label']).sum().item()
            if i % 100 == 0:
                # 输出重建图像：
                save_image(output['decoded'].cpu(), 'results/reconstruction_{}_{}.png'.format(epoch, step))

        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
        writer.add_scalar('accuracy', correct / total, step)

        # 随机取一些看看：
        sample = torch.randn(16, 2048).to(const.device)
        sample = net.decoder(sample).cpu()
        save_image(sample, 'results/sample_{}.png'.format(epoch), nrow=4)
        
if __name__ == '__main__':
    main()
        
