import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from datasets import load_metric

def compute_cer(processor, pred_ids, label_ids):
    cer_metric = load_metric("cer")
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

def evaluate_text_recognizer(model,data_loader,device,processor) :
    losses = []
    valid_cer = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader) :
            for k , v in batch.items() :
                batch[k] = v.to(device)

            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss.item())
            
            outputs = model.generate(batch["pixel_values"].to(device))
            cer = compute_cer(processor=processor,pred_ids=outputs, label_ids=batch["labels"])
            valid_cer.append(cer) 

        avg_loss = np.average(losses)
        validation_cer = np.average(valid_cer)
    return avg_loss , validation_cer