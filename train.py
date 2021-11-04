'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from cbam.imagenet import create_resnet
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from torchsampler import ImbalancedDatasetSampler


def get_WeightedRandom_Sampler(dataset):

    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=512)

    All_target = []
    for _, (_, targets) in enumerate(dataLoader):
        for i in range(targets.shape[0]):
            All_target.append(targets[i].item())

    target = numpy.array(All_target)

    for i in numpy.unique(target):
        print(numpy.sum(target == i))  # 对照unique数组，依次统计每个元素出现的次数

    # 首先统计全部类别的数目
    class_sample_count = numpy.array(
        [len(numpy.where(target == t)[0]) for t in numpy.unique(target)])
    # print(class_sample_count)

    # 计算每个类别的权重
    weight = 1. / class_sample_count
    # print(weight)

    # 计算samples_weight——注意这个值的维度和target的维度一致
    samples_weight = numpy.array([weight[t] for t in target])
    # print(samples_weight.shape)

    # 转化成tensor数据类型
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()

    # 定义sampler
    Sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    return Sampler

def get_dataloader_target_class_number(dataLoader):

    All_target_2 = []
    for batch_idx, (inputs, targets) in enumerate(dataLoader):
        for i in range(targets.shape[0]):
            All_target_2.append(targets[i].item())

    data = numpy.array(All_target_2)

    print('numpy.unique(data)\n', numpy.unique(data))  # unique返回的是已排序数组

    for i in numpy.unique(data):
        print(numpy.sum(data == i))  # 对照unique数组，依次统计每个元素出现的次数

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('-d', '--data', default='skinV2', type=str)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N', help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=64, type=int, metavar='N', help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='')
parser.add_argument('--depth', type=int, default=18, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    import time
    now2 = time.strftime("%m-%d", time.localtime())
    args.checkpoint = 'checkpoint/{}_{}_Resnet{}_{}_{}_{}'.format(
        now2, args.data, (args.depth if args.arch is '' else str(args.depth) +'_' + args.arch), args.epochs, args.train_batch, args.lr)

    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    if args.data == 'diabetic':
        data_dir = '/workspace/cpfs-data/code/medical_image/diabetic'
        num_class = 5
    elif args.data == 'skin':
        data_dir = '/workspace/cpfs-data/code/medical_image/HAM10000'
        num_class = 7
    elif args.data == 'chest':
        data_dir = '/workspace/cpfs-data/code/medical_image/ChestXRay'
        num_class = 15
    

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train':transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize]),
        'val':transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            normalize]),
    }

    image_datasets = datasets.ImageFolder(data_dir,data_transforms['train'])
    train_size = int(0.8 * len(image_datasets))
    val_size = len(image_datasets) - train_size

    train_set, val_set = torch.utils.data.random_split(image_datasets, [train_size, val_size])
    weight_sampler = get_WeightedRandom_Sampler(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=weight_sampler, batch_size=args.train_batch, num_workers=args.workers, pin_memory=True)   
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch, num_workers=args.workers, pin_memory=True)

    get_dataloader_target_class_number(train_loader)
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # train_loader = torch.utils.data.DataLoader(image_datasets['train'], sampler=ImbalancedDatasetSampler(image_datasets['train']), batch_size=args.train_batch, num_workers=args.workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(image_datasets['val'], sampler=ImbalancedDatasetSampler(image_datasets['val']), batch_size=args.test_batch, num_workers=args.workers, pin_memory=True)


    # create model
    model = create_resnet(args.depth, num_class, args.arch)

    model = model.cuda()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = '{}_Resnet{}'.format(args.data, (args.depth if args.arch is '' else str(args.depth) +'_' + args.arch))
    

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.word('DataSet:{}\nModel:Resnet{}\n'.format(args.data,(args.depth if args.arch is '' else str(args.depth) +'_' + args.arch)))
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)
    logger.word('\nBest acc:{}'.format(best_acc))
    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    end = time.time()

    # bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top3: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    ))

    return (losses.avg, top1.avg)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs.data, targets.data, topk=(1, 2))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top3.update(prec3.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f} | top1: {top1: .4f} | top2: {top3: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top3=top3.avg,
                    ))

    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    # if epoch in args.schedule:
    if epoch % 100 == 0 and epoch != 0:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
