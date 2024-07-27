from tqdm import tqdm
import torch
import numpy as np

def eval_detector_epoch(model,data_loader,device) :
    losses = []
    model.train()
    for i in tqdm(data_loader) :
        images = [i["images"][0].to(device).to(torch.float32)]
        boxes = i["targets"]['boxes'][0].to(device).to(torch.int64)
        labels  = i["targets"]['labels'][0].to(device).to(torch.int64)
        targets = [{'boxes':boxes , 'labels' :labels } ]

        output = model(images,targets)

        compined_loss = sum(loss for loss in output.values())
        losses.append(compined_loss.item())

    avg_loss = np.average(losses)
    return avg_loss