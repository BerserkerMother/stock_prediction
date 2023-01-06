import argparse
import logging
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.cuda import amp

from model import Transformer
from data import StockDataset
from utils import set_seed, AverageMeter


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    logging.info("preparing the datasets")
    data_set = StockDataset(args.data)
    total_size = len(data_set)
    train_size = int(total_size * .8)
    val_size = total_size - train_size
    train_set, val_set = random_split(data_set, [train_size, val_size])
    logging.info("%d total samples, %d train samples, %d val samples"
                 % (total_size, train_size, val_size))

    logging.info("initializing the model!")
    model = Transformer(num_heads=args.num_heads,
                        d_model=args.d_model,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        )
    model.to(device)
    optim = Adam(model.parameters(), lr=args.lr,
                 weight_decay=args.weight_decay)

    scaler = amp.GradScaler()
    # loads model.pth
    if os.path.exists("pretrain.pth"):
        if args.use_model:
            state_dict = torch.load("pretrain.pth")
            model.load_state_dict(state_dict, strict=True)
            logging.info("ATTENTION: loaded pretrained model")

    num_parameters = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
    logging.info('model has %d parameters' % num_parameters)

    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        num_workers=4,
        batch_size=args.batch_size
    )
    val_loader = DataLoader(
        dataset=val_set,
        shuffle=True,
        num_workers=4,
        batch_size=args.batch_size
    )

    logging.info('Starting Training...')
    for e in range(args.max_epoch):
        train_loss, train_acc = train_step(model, optim, train_loader, scaler, device)
        val_loss, val_acc = val_step(model, val_loader, device)
        logging.info("Epoch[%3d/%d], train loss: %f, train acc: %2.2f%%,\nval loss: %f, val acc: %2.2f%%" %
                     (e, args.max_epoch, train_loss, train_acc * 100, val_loss, val_acc * 100))
        logging.info(50 * '-')
        # save model's state dict
        torch.save(model.state_dict(), "pretrain.pth")


def train_step(model, optim, ft_train_loader, scaler, device):
    model.train()
    loss_avg, acc_meter = AverageMeter(), AverageMeter()
    for i, (feature, target1, target2) in enumerate(ft_train_loader):
        batch_size = feature.size()[0]
        feature = feature.to(device)
        target1 = target1.to(device).to(torch.float)
        target2 = target2.to(device).to(torch.float)

        with amp.autocast():
            state, reg = model(feature)
            # loss
            state_loss = F.binary_cross_entropy_with_logits(state, target1)
            reg_loss = F.mse_loss(reg, target2)
            loss = state_loss + reg_loss

        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        pred = torch.where(state >= .5, 1, 0)
        num_correct = (pred == target1).sum()
        acc_meter.update(num_correct, batch_size)

        loss_avg.update(loss.item(), 1)

    return loss_avg.avg(), acc_meter.avg()


@torch.no_grad()
def val_step(model, ft_test_loader, device):
    model.eval()
    loss_avg = AverageMeter()
    acc_meter = AverageMeter()
    for i, (feature, target1, target2) in enumerate(ft_test_loader):
        batch_size = feature.size()[0]
        feature = feature.to(device)
        target1 = target1.to(device)
        target2 = target2.to(device)

        state, reg = model(feature)
        # loss
        state_loss = F.binary_cross_entropy_with_logits(state, target1)
        reg_loss = F.mse_loss(reg, target2)
        loss = state_loss + reg_loss

        pred = torch.where(state >= .5, 1, 0)
        num_correct = (pred == target1).sum()
        acc_meter.update(num_correct, batch_size)
        loss_avg.update(loss.item(), 1)

    return loss_avg.avg(), acc_meter.avg()


arg_parser = argparse.ArgumentParser('transformer for stock prediction')
arg_parser.add_argument('--num_heads', default=4, type=int,
                        help="number of self attention heads")
arg_parser.add_argument('--d_model', default=512, type=int)
arg_parser.add_argument('--num_layers', default=3, type=int,
                        help="number of transformer layers")
arg_parser.add_argument('--dropout', default=0.3, type=float,
                        help="model dropout")

arg_parser.add_argument('--lr', default=1e-4, type=float, help="optimizer lr")
arg_parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help="optimizer l2 norm weight")

arg_parser.add_argument('--data', type=str, required=True, help="path to data folder")
arg_parser.add_argument('--batch_size', default=4, type=int,
                        help="mini batch size")
arg_parser.add_argument('--max_epoch', default=500, type=int,
                        help="number of training epochs")
arg_parser.add_argument("--save", action="store_true",
                        help="if true it saved model after each epoch")
arg_parser.add_argument("--use_model", action="store_true",
                        help="use pretrained model")
arg_parser.add_argument("--seed", default=123, type=int,
                        help="seed for randomness")

arguments = arg_parser.parse_args()

logging.basicConfig(
    format='[%(levelname)s] %(module)s - %(message)s',
    level=logging.INFO
)

if __name__ == '__main__':
    main(arguments)
