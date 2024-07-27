import torch
import torch.nn as nn
import torchvision.models as models
from transformers import VisionEncoderDecoderModel
    
def get_text_recognizer_model(processor) :
    text_recognizer_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    # set special tokens used for creating the decoder_input_ids from the labels
    text_recognizer_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    text_recognizer_model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    text_recognizer_model.config.vocab_size = text_recognizer_model.config.decoder.vocab_size

    # set beam search parameters
    text_recognizer_model.config.eos_token_id = processor.tokenizer.sep_token_id
    text_recognizer_model.config.max_length = 64
    text_recognizer_model.config.early_stopping = True
    text_recognizer_model.config.no_repeat_ngram_size = 3
    text_recognizer_model.config.length_penalty = 2.0
    text_recognizer_model.config.num_beams = 4

    return text_recognizer_model