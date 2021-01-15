import os, sys
sys.path.append('../')
os.chdir('../')

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from transformers import BertTokenizer, BertConfig, BertForPreTraining
from nltk.tokenize import TweetTokenizer, word_tokenize

from modules.multi_label_classification import BertForMultiLabelClassification
from utils.forward_fn import forward_sequence_multi_classification
from utils.metrics import absa_metrics_fn
from utils.data_utils import AspectBasedSentimentAnalysisProsaDataset, AspectBasedSentimentAnalysisDataLoader







###
# common functions
###
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)






# Set random seed
set_seed(26092020)








tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
config.num_labels = max(AspectBasedSentimentAnalysisProsaDataset.NUM_LABELS)
config.num_labels_list = AspectBasedSentimentAnalysisProsaDataset.NUM_LABELS

# Instantiate model
model = BertForMultiLabelClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)





model
print(model)

print(count_param(model))

#
# test = './dataset/casa_absa-prosa/vocab.txt'
# print(test)
train_dataset_path = './dataset/casa_absa-prosa/train_preprocess.csv'
valid_dataset_path = './dataset/casa_absa-prosa/valid_preprocess.csv'
test_dataset_path = './dataset/casa_absa-prosa/test_preprocess_masked_label.csv'



train_dataset = AspectBasedSentimentAnalysisProsaDataset(train_dataset_path, tokenizer, lowercase=True)
valid_dataset = AspectBasedSentimentAnalysisProsaDataset(valid_dataset_path, tokenizer, lowercase=True)
test_dataset = AspectBasedSentimentAnalysisProsaDataset(test_dataset_path, tokenizer, lowercase=True)

train_loader = AspectBasedSentimentAnalysisDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=16, num_workers=16, shuffle=True)
valid_loader = AspectBasedSentimentAnalysisDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=16, num_workers=16, shuffle=False)
test_loader = AspectBasedSentimentAnalysisDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=16, num_workers=16, shuffle=False)



w2i, i2w = AspectBasedSentimentAnalysisProsaDataset.LABEL2INDEX, AspectBasedSentimentAnalysisProsaDataset.INDEX2LABEL
print(w2i)
print(i2w)



# TEST MODEL ON SAMPLE SENTENCES 1

print('--------------------------')
print('Test sample sentences 1')
print('--------------------------')
text = 'mesin 3SZ - VE 1500 cc ini memang lebih pas di badan Avanza, bertenaga namun hemat bahan bakar'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

print(f'Text: {text}')
for i, label in enumerate(labels):
    print(f'Label `{AspectBasedSentimentAnalysisProsaDataset.ASPECT_DOMAIN[i]}` : {i2w[label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')



# TEST MODEL ON SAMPLE SENTENCES 2

print('--------------------------')
print('Test sample sentences 2')
print('--------------------------')
text = 'Jazz enak. tetapi mahal harga dan perawatan gila .'
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

print(f'Text: {text}')
for i, label in enumerate(labels):
    print(f'Label `{AspectBasedSentimentAnalysisProsaDataset.ASPECT_DOMAIN[i]}` : {i2w[label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')



# TEST MODEL ON SAMPLE SENTENCES 3
print('--------------------------')
print('Test sample sentences 3')
print('--------------------------')
text = 'Toyota sekarang harga kemahalan, kualitas menurun, servicenya parah, dan boros bensin. '
subwords = tokenizer.encode(text)
subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)

logits = model(subwords)[0]
labels = [torch.topk(logit, k=1, dim=-1)[1].squeeze().item() for logit in logits]

print(f'Text: {text}')
for i, label in enumerate(labels):
    print(f'Label `{AspectBasedSentimentAnalysisProsaDataset.ASPECT_DOMAIN[i]}` : {i2w[label]} ({F.softmax(logits[i], dim=-1).squeeze()[label] * 100:.3f}%)')


# Fine Tuning & Evaluation
optimizer = optim.Adam(model.parameters(), lr=5e-6)
# model = model.cuda()
model = model.cpu()



# TRAIN
print('--------------------------')
print('TRAINING')
print('--------------------------')
n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    torch.set_grad_enabled(True)

    total_train_loss = 0
    list_hyp, list_label = [], []

    train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
    for i, batch_data in enumerate(train_pbar):
        # Forward model
        loss, batch_hyp, batch_label = forward_sequence_multi_classification(model, batch_data[:-1], i2w=i2w,
                                                                             device='cuda')

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tr_loss = loss.item()
        total_train_loss = total_train_loss + tr_loss

        # Calculate metrics
        list_hyp += batch_hyp
        list_label += batch_label

        train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch + 1),
                                                                                   total_train_loss / (i + 1),
                                                                                   get_lr(optimizer)))

    # Calculate train metric
    metrics = absa_metrics_fn(list_hyp, list_label)
    print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch + 1),
                                                             total_train_loss / (i + 1), metrics_to_string(metrics),
                                                             get_lr(optimizer)))

    # Evaluate on validation
    model.eval()
    torch.set_grad_enabled(False)

    total_loss, total_correct, total_labels = 0, 0, 0
    list_hyp, list_label = [], []

    pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]
        loss, batch_hyp, batch_label = forward_sequence_multi_classification(model, batch_data[:-1], i2w=i2w,
                                                                             device='cuda')

        # Calculate total loss
        valid_loss = loss.item()
        total_loss = total_loss + valid_loss

        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label
        metrics = absa_metrics_fn(list_hyp, list_label)

        pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss / (i + 1), metrics_to_string(metrics)))

    metrics = absa_metrics_fn(list_hyp, list_label)
    print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch + 1), total_loss / (i + 1), metrics_to_string(metrics)))