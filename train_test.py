import torch
import torch.nn.functional as F

import numpy as np
def calculate_metrics(outputs,labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for pred, act in zip(outputs, labels):
        if pred == 1 and act == 1:
            tp += 1
        elif pred == 1 and act == 0:
            fp += 1
        elif pred == 0 and act == 0:
            tn += 1
        else:
            fn += 1

    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0

    return precision,recall



def train_model(model, device, train_dataloader, optimizer, noise_factor, epoch):
    MSE = []
    model.train()

    for batch_index,(data,target) in enumerate(train_dataloader):

        data = data.view(data.size(0), -1)
        data = data.to(device)
        data_addnoise = data + noise_factor * torch.randn_like(data)
        data_addnoise = data_addnoise.to(device)
        optimizer.zero_grad()
        output = model(data_addnoise)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        MSE.append(loss)

    print(f"TRAIN_EPOCH:[{epoch}] MSE:{sum(MSE)/len(MSE):.4f}")
    return sum(MSE) / len(MSE)


def test_model(model, device, test_dataloader, noise_factor, epoch):
    MSE = []
    with torch.no_grad():
        for batch_index, (data, _) in enumerate(test_dataloader):
            data = data.view(data.size(0), -1)
            data = data.to(device)
            data_addnoise = data + noise_factor * torch.randn_like(data)
            data_addnoise = data_addnoise.to(device)
            output = model(data_addnoise)
            loss = F.mse_loss(output, data)
            MSE.append(loss)
    print(f"TEST__EPOCH:[{epoch}] MSE:{sum(MSE)/len(MSE):.4f}")
    return sum(MSE) / len(MSE)



def train_cnnmodel(model, device, train_dataloader, optimizer, epoch):
    MSE = []
    model.train()

    for batch_index,(data,target) in enumerate(train_dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        MSE.append(loss)

    print(f"TRAIN_EPOCH:[{epoch}] MSE:{sum(MSE)/len(MSE):.4f}")
    return sum(MSE) / len(MSE)


def test_cnnmodel(model, device, test_dataloader, epoch):
    MSE = []
    with torch.no_grad():
        for batch_index, (data, _) in enumerate(test_dataloader):
            data = data.to(device)

            output = model(data)
            loss = F.mse_loss(output, data)
            MSE.append(loss)
    print(f"TEST__EPOCH:[{epoch}] MSE:{sum(MSE)/len(MSE):.4f}")
    return sum(MSE) / len(MSE)
###################################################################
def train_new_model(model, device, train_dataloader, optimizer, noise_factor, epoch):
    LOSS = []
    model.train()
    P = []
    R = []
    for batch_index,(data,target) in enumerate(train_dataloader):
        data_p = []
        data_true = []
        pre = []
        true = []
        data = data.view(data.size(0), -1)
        data, target = data.to(device), target.to(device)
        data_addnoise = data + noise_factor * torch.randn_like(data)
        data_addnoise = data_addnoise.to(device)
        optimizer.zero_grad()
        output = model(data_addnoise)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        LOSS.append(loss)
        _, predicted = torch.max(output.data, 1)
        data_p.extend(predicted.cpu().numpy())
        data_true.extend(target.cpu().numpy())
        for i in range(10):
            pre.append([1 if j == i else 0 for j in data_p])
            true.append([1 if j == i else 0 for j in data_true])
            prec, reca = calculate_metrics(pre[i], true[i])

            P.append(prec)
            R.append(reca)
    Precison = np.mean(np.array(P))
    Recall = np.mean(np.array(R))

    print(f"TRAIN_EPOCH:[{epoch}] Loss:{(sum(LOSS)/len(LOSS)):.4f} Precision:{Precison} Recall:{Recall}")
    return Precison, Recall


def test_new_model(model, device, test_dataloader, noise_factor, epoch):
    LOSS = []
    P = []
    R = []
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(test_dataloader):
            data_p = []
            data_true = []
            pre = []
            true = []
            data = data.view(data.size(0), -1)
            data, target = data.to(device), target.to(device)
            data_addnoise = data + noise_factor * torch.randn_like(data)
            data_addnoise = data_addnoise.to(device)
            output = model(data_addnoise)
            loss = F.cross_entropy(output,target)
            LOSS.append(loss)
        _, predicted = torch.max(output.data, 1)
        data_p.extend(predicted.cpu().numpy())
        data_true.extend(target.cpu().numpy())
        for i in range(10):
            pre.append([1 if j == i else 0 for j in data_p])
            true.append([1 if j == i else 0 for j in data_true])
            prec, reca = calculate_metrics(pre[i], true[i])
            P.append(prec)
            R.append(reca)

    Precison = np.mean(np.array(P))
    Recall = np.mean(np.array(R))
    print(f"TEST__EPOCH:[{epoch}] Loss:{sum(LOSS)/len(LOSS):.4f} Precision:{Precison} Recall:{Recall}")
    return Precison, Recall
################################
def train_new_cnnmodel(model, device, train_dataloader, optimizer, noise_factor, epoch):
    LOSS = []
    model.train()
    P = []
    R = []
    for batch_index,(data,target) in enumerate(train_dataloader):
        data_p = []
        data_true = []
        pre = []
        true = []
        data, target = data.to(device), target.to(device)
        data_addnoise = data + noise_factor * torch.randn_like(data)
        data_addnoise = data_addnoise.to(device)
        optimizer.zero_grad()
        output = model(data_addnoise)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        LOSS.append(loss)
        _, predicted = torch.max(output.data, 1)
        data_p.extend(predicted.cpu().numpy())
        data_true.extend(target.cpu().numpy())
        for i in range(10):
            pre.append([1 if j == i else 0 for j in data_p])
            true.append([1 if j == i else 0 for j in data_true])
            prec, reca = calculate_metrics(pre[i], true[i])

            P.append(prec)
            R.append(reca)
    Precison = np.mean(np.array(P))
    Recall = np.mean(np.array(R))

    print(f"TRAIN_EPOCH:[{epoch}] Loss:{(sum(LOSS)/len(LOSS)):.4f} Precision:{Precison} Recall:{Recall}")
    return Precison, Recall


def test_new_cnnmodel(model, device, test_dataloader, noise_factor, epoch):
    LOSS = []
    P = []
    R = []
    with torch.no_grad():
        for batch_index, (data, target) in enumerate(test_dataloader):
            data_p = []
            data_true = []
            pre = []
            true = []
            data, target = data.to(device), target.to(device)
            data_addnoise = data + noise_factor * torch.randn_like(data)
            data_addnoise = data_addnoise.to(device)
            output = model(data_addnoise)
            loss = F.cross_entropy(output,target)
            LOSS.append(loss)
        _, predicted = torch.max(output.data, 1)
        data_p.extend(predicted.cpu().numpy())
        data_true.extend(target.cpu().numpy())
        for i in range(10):
            pre.append([1 if j == i else 0 for j in data_p])
            true.append([1 if j == i else 0 for j in data_true])
            prec, reca = calculate_metrics(pre[i], true[i])
            P.append(prec)
            R.append(reca)

    Precison = np.mean(np.array(P))
    Recall = np.mean(np.array(R))
    print(f"TEST__EPOCH:[{epoch}] Loss:{sum(LOSS)/len(LOSS):.4f} Precision:{Precison} Recall:{Recall}")
    return Precison, Recall


