import time

import matplotlib.pyplot as plt
import torch


def loss_graph(losses, net_list):
    for experiment_id in net_list:
        plt.plot(losses[experiment_id], label=experiment_id)

    plt.legend()
    plt.title('Loss')
    plt.tight_layout()


def train(net, train_dl, epoch_num=200, epoch_info_show=20, weight_decay=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    loss = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3, weight_decay=weight_decay)

    t = time.time()
    loss_history = []

    for epoch in range(1, epoch_num + 1):

        total_loss = 0
        for points_b, sdfs_b in train_dl:
            points_b = points_b.to(device)
            sdfs_b = sdfs_b.to(device)

            pred = net.forward(points_b)
            pred = pred.squeeze()
            loss_value = loss(pred, sdfs_b)
            total_loss += loss_value
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_history.append(total_loss.item())
        if epoch % epoch_info_show == 0:
            time_ = time.strftime("%H:%M:%S", time.gmtime(time.time() - t))
            print(f' Train Epoch: {epoch} Time: {time_} Loss: {round(total_loss.item(), 4)}')

    del net
    return loss_history
