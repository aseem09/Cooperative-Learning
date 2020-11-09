import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--c_lambda', type=float, default=1e-1,
                    help='cooperative loss coefficient')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

# Loss Function of one Model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model_1 = Network(args.init_channels, CIFAR_CLASSES,
                      args.layers, criterion)
    model_2 = Network(args.init_channels, CIFAR_CLASSES,
                      args.layers, criterion)

    model_1 = model_1.cuda()
    model_2 = model_2.cuda()

    # utils.load(model_1, '/content/search-EXP-20201108-231407/weights1.pt')
    # utils.load(model_2, '/content/search-EXP-20201108-231407/weights2.pt')

    logging.info("Param size Model 1 = %fMB",
                 utils.count_parameters_in_MB(model_1))
    logging.info("Param size Model 2 = %fMB",
                 utils.count_parameters_in_MB(model_2))

    optimizer_1 = torch.optim.SGD(
        model_1.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer_2 = torch.optim.SGD(
        model_2.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_1, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model_1, model_2, args.c_lambda, args)

    for epoch in range(args.epochs):
        scheduler.step()

        lr = scheduler.get_lr()[0]

        logging.info('Epoch %d lr %e', epoch, lr)
        # logging.info('Epoch %d lr %e', epoch, lr_2)

        genotype_1 = model_1.genotype()
        genotype_2 = model_2.genotype()

        logging.info('Genotype Model 1 = %s', genotype_1)
        logging.info('Genotype Model 2 = %s', genotype_2)

        print(F.softmax(model_1.alphas_normal, dim=-1))
        print(F.softmax(model_1.alphas_reduce, dim=-1))

        print(F.softmax(model_2.alphas_normal, dim=-1))
        print(F.softmax(model_2.alphas_reduce, dim=-1))

        # training
        train_acc_1, train_acc_2, train_obj = train(
            train_queue, valid_queue, model_1, model_2, architect, criterion, optimizer_1, optimizer_2, lr)
        logging.info('Train_Acc Model 1 %f', train_acc_1)
        logging.info('Train_Acc Model 2 %f', train_acc_2)

        # validation
        valid_acc_1, valid_obj_1, valid_acc_2, valid_obj_2 = infer(
            valid_queue, model_1, model_2, criterion)
        logging.info('Valid_Acc Model 1 %f', valid_acc_1)
        logging.info('Valid_Acc Model 2 %f', valid_acc_2)

        utils.save(model_1, os.path.join(args.save, 'weights1.pt'))
        utils.save(model_2, os.path.join(args.save, 'weights2.pt'))


def train(train_queue, valid_queue, model_1, model_2, architect, criterion, optimizer_1, optimizer_2, lr):
    objs = utils.AvgrageMeter()

    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()

    top1_2 = utils.AvgrageMeter()
    top5_2 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model_1.train()
        model_2.train()

        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(
            target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search,
                       lr, optimizer_1, optimizer_2, unrolled=args.unrolled)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        logits_1 = model_1(input)
        logits_2 = model_2(input)

        loss = architect._get_loss_val(input, target)
        loss.backward()

        nn.utils.clip_grad_norm_(model_1.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(model_2.parameters(), args.grad_clip)
        optimizer_1.step()
        optimizer_2.step()

        prec1_1, prec5_1 = utils.accuracy(logits_1, target, topk=(1, 5))
        prec1_2, prec5_2 = utils.accuracy(logits_2, target, topk=(1, 5))
        objs.update(loss.item(), n)

        top1_1.update(prec1_1.item(), n)
        top5_1.update(prec5_1.item(), n)

        top1_2.update(prec1_2.item(), n)
        top5_2.update(prec5_2.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train %03d %f %e %f %f %f %f', step, loss, objs.avg,
                         top1_1.avg, top5_1.avg,  top1_2.avg, top5_2.avg)

    return top1_1.avg, top1_2.avg, objs.avg


def infer(valid_queue, model_1, model_2, criterion):
    objs_1 = utils.AvgrageMeter()
    objs_2 = utils.AvgrageMeter()

    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()

    top1_2 = utils.AvgrageMeter()
    top5_2 = utils.AvgrageMeter()

    model_1.eval()
    model_2.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(non_blocking=True)

        logits_1 = model_1(input)
        logits_2 = model_2(input)

        loss_1 = criterion(logits_1, target)
        loss_2 = criterion(logits_2, target)

        prec1_1, prec5_1 = utils.accuracy(logits_1, target, topk=(1, 5))
        prec1_2, prec5_2 = utils.accuracy(logits_2, target, topk=(1, 5))

        n = input.size(0)
        objs_1.update(loss_1.item(), n)
        objs_2.update(loss_2.item(), n)

        top1_1.update(prec1_1.item(), n)
        top5_1.update(prec5_1.item(), n)

        top1_2.update(prec1_2.item(), n)
        top5_2.update(prec5_2.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid %03d %e %e %f %f %f %f', step,
                         objs_1.avg, objs_2.avg, top1_1.avg, top5_1.avg, top1_2.avg, top5_2.avg)

    return top1_1.avg, objs_1.avg, top1_2.avg, objs_2.avg


if __name__ == '__main__':
    main()
