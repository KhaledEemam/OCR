from tqdm import tqdm
import torch
import numpy as np

def train_detector_epoch(model,optimizer,scheduler,data_loader,device) :
    losses = []
    model.train()
    for i in tqdm(data_loader) :
        optimizer.zero_grad()
        images = [i["images"][0].to(device).to(torch.float32)]
        boxes = i["targets"]['box'][0].to(device).to(torch.int64)
        labels  = i["targets"]['label'][0].to(device).to(torch.int64)
        targets = [{'boxes':boxes , 'labels' :labels } ]

        output = model(images,targets)
        compined_loss = sum(loss for loss in output.values())
        losses.append(compined_loss.item())
        compined_loss.backward()
        optimizer.step()
        scheduler.step()
    
    avg_loss = np.average(losses)
    
    return avg_loss