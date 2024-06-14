import os

import torch
import torch.nn as nn
from tqdm import tqdm
from os.path import join
from dataset import get_loader
from parse_args import get_args
from models.net import get_model
from tensorboardX import SummaryWriter


@torch.enable_grad()
def train(net, loss_fn, optimizer, loader, writer, epoch, args):
    net.train()
    pbar = tqdm(total=len(loader), desc=f"{args.task} ({args.arch}) train epoch {epoch}")
    loss_his = []
    total = 0
    pred_corrects = 0
    for batch_idx, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        images, labels = images.to(args.device), labels.to(args.device)
        y_pred = net(images)
        loss = loss_fn(y_pred, labels)
        loss_his.append(loss.item())
        loss.backward()
        optimizer.step()
        total += images.size(0)
        _, pred_cls = torch.max(y_pred, dim=-1)
        pred_corrects += torch.sum(torch.eq(pred_cls, labels)).item()
        pbar.update(1)
        pbar.set_postfix_str(
            f"loss: {loss_his[-1]:.4f}, acc: {pred_corrects / total:.2%}, best_eval_acc:{args.top_acc:.2%}")
        writer.add_scalar('iter_loss', loss_his[-1], global_step=batch_idx + 1 + len(loader) * epoch)
    pbar.close()
    avg_loss = sum(loss_his) / len(loss_his)
    writer.add_scalar('avg_epoch_loss', avg_loss, global_step=epoch)
    writer.add_scalar('train_epoch_acc', pred_corrects / total, global_step=epoch)
    return avg_loss


@torch.no_grad()
def evaluate(net, loader, writer, epoch, args) -> float:
    net.eval()
    pbar = tqdm(total=len(loader), desc=f"{args.task} ({args.arch}) evaluate epoch {epoch}")
    total = 0
    pred_corrects = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(args.device), labels.to(args.device)
        y_pred = net(images)
        total += images.size(0)
        _, pred_cls = torch.max(y_pred, dim=-1)
        pred_corrects += torch.sum(torch.eq(pred_cls, labels)).item()
        pbar.update(1)
        pbar.set_postfix_str(f"ACC current/best: {pred_corrects / total:.2%}/{args.top_acc:.2%}")
    pbar.close()
    acc = pred_corrects / total
    writer.add_scalar('evaluate_epoch_acc', acc, global_step=epoch)
    return acc


def main():
    args = get_args()
    # fix the random seed for reproducibility
    torch.manual_seed(args.random_seed)
    net = get_model(arch=args.arch, num_classes=args.num_classes, use_pretrained=args.use_pretrained).to(args.device)
    train_dataloader = get_loader(label_path=args.train_label_path,
                                  data_root=args.data_root,
                                  task=args.task,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    eval_dataloader = get_loader(label_path=args.eval_label_path,
                                 data_root=args.data_root,
                                 task=args.task,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 drop_last=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(logdir=args.log_dir)
    args.top_acc = 0
    best_acc_path = None
    for epoch in range(1, args.epochs + 1):
        avg_train_loss = train(net=net, loss_fn=loss_fn, optimizer=optimizer, loader=train_dataloader,
                               writer=writer, epoch=epoch, args=args)
        acc = evaluate(net=net, loader=eval_dataloader, writer=writer, epoch=epoch, args=args)
        if acc >= args.top_acc:
            args.top_acc = acc
            if args.ckpt_dir:
                if best_acc_path is not None:
                    os.remove(best_acc_path)
                best_acc_path = join(args.log_dir, f"best-{args.arch}_epoch-{epoch}_acc-{acc * 100:.2f}.pth")
                torch.save(net.state_dict(), best_acc_path)
        if epoch % args.ckpt_interval == 0:
            if args.ckpt_dir:
                to_save = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(to_save, join(args.ckpt_dir, f"{args.arch}_epoch-{epoch}_acc-{acc * 100:.2f}.ckpt"))
        if avg_train_loss <= args.eps:
            print(f'Training early stop at epoch {epoch}, '
                  f'because the average epoch loss {avg_train_loss} ≤ {args.eps}.')
            break
    print(f"Finished training, the best accuracy is {args.top_acc:.2%}.")


if __name__ == "__main__":
    main()
