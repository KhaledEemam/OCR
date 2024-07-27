import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def train_text_recognizer(model,data_loader,optimizer,device) :
    losses = []
    model.train()

    for batch in tqdm(data_loader) :
        optimizer.zero_grad()
        for k , v in batch.items() :
            batch[k] = v.to(device)

        outputs = model(**batch)
        loss = outputs.loss
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    avg_loss = np.average(losses)
    return avg_loss