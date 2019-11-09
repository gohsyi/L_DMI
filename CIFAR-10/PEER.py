# CE - training with cross entropy loss

import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import *
from dataset import *

num_classes = 10

CE = nn.CrossEntropyLoss().cuda()


def train(train_loader, peer_data_loader, peer_labels_loader, model, alpha, optimizer, criterion=CE):
    model.train()

    for i, ((idx, input, target), (_, peer_input, _), (_, _, peer_target)) in \
            enumerate(zip(train_loader, peer_data_loader, peer_labels_loader)):

        if idx.size(0) != batch_size:
            continue

        input = torch.autograd.Variable(input.cuda())
        target = torch.autograd.Variable(target.cuda())
        peer_input = torch.autograd.Variable(peer_input.cuda())
        peer_target = torch.autograd.Variable(peer_target.cuda())

        output = model(input)
        peer_output = model(peer_input)
        loss = criterion(output, target) - criterion(peer_output, peer_target) * alpha

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, test_loader=test_loader_):
    model.eval()
    correct = 0
    total = 0

    for i, (idx, input, target) in enumerate(tqdm(test_loader)):
        input = torch.Tensor(input).cuda()
        target = torch.autograd.Variable(target).cuda()

        total += target.size(0)
        output = model(input)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total

    return accuracy


def main_peer(alpha):
    model_peer = torch.load('./model_ce_' + str(args.r) + '_' + str(args.s))
    best_valid_loss = test(test_loader=valid_loader_noisy, model=model_peer, criterion=CE)
    torch.save(model_peer, './model_peer_' + str(args.r) + '_' + str(args.s))
    test_acc = test(model=model_peer, test_loader=test_loader_)
    print('test_acc=', test_acc)

    for epoch in range(100):
        print("epoch=", epoch,'r=', args.r)

        learning_rate = 1e-6
        optimizer_peer = torch.optim.SGD(model_peer.parameters(), momentum=0.9, weight_decay=1e-4, lr=learning_rate)
        #print("traning model_ce...")
        train(train_loader=train_loader_noisy, peer_data_loader=train_loader_noisy_peer_data, alpha=alpha,
              peer_labels_loader=train_loader_noisy_peer_labels, model=model_ce, optimizer=optimizer_ce)
        #print("validating model_ce...")
        valid_acc = test(model=model_peer, test_loader=valid_loader_noisy)
        print('valid_acc=', valid_acc)
        if valid_acc >= best_peer_acc:
            best_peer_acc = valid_acc
            torch.save(model_peer, './model_peer_' + str(args.r) + '_' + str(args.s) + str(alpha))
            print("saved.")


def evaluate(path):
    model = torch.load(path)
    test_acc = test(model=model, test_loader=test_loader_)
    print('test_acc=', test_acc)


if __name__ == '__main__':
    for alpha in [1.0, -5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2.0, 3.0, 4.0, 5.0]:
        main_peer(alpha)
        print("PEER traning finished")
        evaluate('./model_peer_' + str(args.r) + '_' + str(args.s) + '_' + str(alpha))

