import os
import transformers
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial
from transformers import get_linear_schedule_with_warmup, AutoConfig 
from transformers import BartTokenizer,BartModel,BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import BartConfig
from transformers import AutoTokenizer
from transformers import AdamW
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import socket
from os.path import basename
import re
import itertools
import json
from pathlib import Path
import glob
import sys
import subprocess
import argparse
from functools import reduce
from pathlib import Path


torch.manual_seed(0)



class TextToGraphQLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, tokenizer, type_path='train.json', block_size=102):
        'Initialization'
        super(TextToGraphQLDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        self.schema_ids = []
        root_path = './SPEGQL-dataset/'
        dataset_path = root_path + 'dataset/' + type_path

        schemas_path = root_path + 'Schemas/'
        schemas = glob.glob(schemas_path + '**/' + 'simpleSchema.json')

        self.max_len = 0
        self.name_to_schema = {}
        for schema_path in schemas:
           with open(schema_path, 'r', encoding='utf-8') as s:
            
             data = json.load(s)

             type_field_tokens = [ ['<t>'] + [t['name']] + ['{'] + [ f['name'] for f in t['fields']] + ['}'] + ['</t>'] for t in data['types']]
             type_field_flat_tokens = reduce(list.__add__, type_field_tokens)

             arguments = [a['name']  for a in data['arguments']]
             schema_tokens = type_field_flat_tokens + ['<a>'] + arguments + ['</a>']

             path = Path(schema_path)
             schema_name = basename(str(path.parent))

             self.name_to_schema[schema_name] = schema_tokens

        with open(dataset_path, 'r', encoding='utf-8') as f:
          data = json.load(f)

          for element in data:
            question_with_schema = 'translate English to GraphQL: ' + element['question']  + ' ' + ' '.join(self.name_to_schema[element['schemaId']])
            tokenized_s = tokenizer.encode_plus(question_with_schema,max_length=1024, padding=True, truncation=True, return_tensors='pt')
            self.source.append(tokenized_s)

            tokenized_t = tokenizer.encode_plus(element['query'],max_length=block_size, padding='max_length', truncation=True, return_tensors='pt')
            self.target.append(tokenized_t)
            self.schema_ids.append(element['schemaId'])

  def get_question_with_schema(self, question, schemaId):
        return 'translate English to GraphQL: ' + question  + ' ' + ' '.join(self.name_to_schema[schemaId])

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

  def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]['input_ids'].squeeze()
        target_ids = self.target[index]['input_ids'].squeeze()
        src_mask = self.source[index]['attention_mask'].squeeze()

        return { 
            'source_ids': source_ids,
                'source_mask': src_mask,
                'target_ids': target_ids,
                'target_ids_y': target_ids
                }