import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup, AutoConfig 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import AdamW
from torch.autograd import Variable
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pandas as pd
import torch.nn.functional as F
import os
import glob
import json
from pathlib import Path
import re
from os.path import basename
from transformers import BartConfig
from functools import reduce
from graphqlval import exact_match
import itertools
torch.manual_seed(0)

"""# Prepare GraphQL Dataset"""

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
        # schemas = glob.glob(schemas_path + '**/' + 'schema.graphql')
        schemas = glob.glob(schemas_path + '**/' + 'simpleSchema.json')

        self.max_len = 0
        self.name_to_schema = {}
        for schema_path in schemas:
           with open(schema_path, 'r') as s:
             data = json.load(s)

             type_field_tokens = [ ['<t>'] + [t['name']] + ['{'] + [ f['name'] for f in t['fields']] + ['}'] + ['</t>'] for t in data['types']]
             type_field_flat_tokens = reduce(list.__add__, type_field_tokens)

             arguments = [a['name']  for a in data['arguments']]
             schema_tokens = type_field_flat_tokens + ['<a>'] + arguments + ['</a>']

             path = Path(schema_path)
             schema_name = basename(str(path.parent))

             self.name_to_schema[schema_name] = schema_tokens

        with open(dataset_path, 'r') as f:
          data = json.load(f)

          for element in data:
            question_with_schema = 'translate English to GraphQL: ' + element['question']  + ' ' + ' '.join(self.name_to_schema[element['schemaId']]) + ' </s>'
            # print(question_with_schema)
            tokenized_s = tokenizer.encode_plus(question_with_schema,max_length=1024, pad_to_max_length=True, truncation=True, return_tensors='pt')
            self.source.append(tokenized_s)

            tokenized_t = tokenizer.encode_plus(element['query'] + ' </s>',max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt')
            self.target.append(tokenized_t)
            self.schema_ids.append(element['schemaId'])

  def get_question_with_schema(self, question, schemaId):
        return 'translate English to GraphQL: ' + question  + ' ' + ' '.join(self.name_to_schema[schemaId]) + ' </s>'


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

class MaskGraphQLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, tokenizer, type_path='train.json', block_size=64):
        'Initialization'
        super(MaskGraphQLDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        path = './SPEGQL-dataset/dataset/' + type_path
        with open(path, 'r') as f:
          data = json.load(f)
          # for element in data:
          for example in data:
            utterance = example['query']
            # tokens = utterance.split()
            encoded_source = tokenizer.encode(utterance + ' </s>', max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt').squeeze()
            token_count = encoded_source.shape[0]
            # print(encoded_source.shape)
            repeated_utterance = [encoded_source for _ in range(token_count)]
            for pos in range(1, token_count):
              encoded_source = repeated_utterance[pos].clone()
              target_id = encoded_source[pos].item()
              if target_id == tokenizer.eos_token_id:
                break
              encoded_source[pos] = tokenizer.mask_token_id
              decoded_target = ''.join(tokenizer.convert_ids_to_tokens([target_id])) + ' </s>'
              encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4, pad_to_max_length=True, truncation=True).squeeze() # should always be of size 1
              self.target.append(encoded_target)
              self.source.append(encoded_source)

              # repeated_utterance[pos][pos] = target_token # so that the next iteration the previous token is correct

                
          

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

  def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]#['input_ids'].squeeze()
        target_id = self.target[index]#['input_ids'].squeeze()
        # src_mask = self.source[index]['attention_mask'].squeeze()
        return { 'source_ids': source_ids,
                'target_id': target_id}

"""# Prepare Spider Dataset"""

class SpiderDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, tokenizer, type_path='train_spider.json', block_size=102):
        'Initialization'
        super(SpiderDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        spider_path = './spider/'
        path = spider_path + type_path

        tables_path = spider_path + 'tables.json'

        with open(path, 'r') as f, open(tables_path, 'r') as t:
          databases = json.load(t)
          data = json.load(f)

          #groupby db_id 
          grouped_dbs = {}
          for db in databases:
            grouped_dbs[db['db_id']] = db
          # print(grouped_dbs)
          # end grop tables

          for element in data:
            db = grouped_dbs[element['db_id']]

            # tables_names = " ".join(db['table_names_original'])
            db_tables = db['table_names_original']

            # columns_names = " ".join([column_name[1] for column_name in db['column_names_original'] ])
            tables_with_columns = ''
            for table_id, group in itertools.groupby(db['column_names_original'], lambda x: x[0]):
              if table_id == -1:
                continue

              columns_names = " ".join([column_name[1] for column_name in group ])
              tables_with_columns += '<t> ' + db_tables[table_id] + ' <c> ' + columns_names + ' </c> ' + '</t> '

            db_with_question = 'translate English to SQL: ' + element['question'] + ' ' + tables_with_columns + '</s>'
            tokenized_s = tokenizer.batch_encode_plus([db_with_question],max_length=1024, pad_to_max_length=True, truncation=True,return_tensors='pt')
            self.source.append(tokenized_s)

            tokenized_t = tokenizer.batch_encode_plus([element['query'] + ' </s>'],max_length=block_size, pad_to_max_length=True, truncation=True,return_tensors='pt')
            self.target.append(tokenized_t)


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

  def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]['input_ids'].squeeze()
        target_ids = self.target[index]['input_ids'].squeeze()
        src_mask = self.source[index]['attention_mask'].squeeze()
        return { 'source_ids': source_ids,
                'source_mask': src_mask,
                'target_ids': target_ids,
                'target_ids_y': target_ids}

class CoSQLMaskDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, tokenizer, type_path='cosql_train.json', block_size=64):
        'Initialization'
        super(CoSQLMaskDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        path = './cosql_dataset/sql_state_tracking/' + type_path
        with open(path, 'r') as f:
          data = json.load(f)
          for element in data:
            for interaction in element['interaction']:
              utterance = interaction['query']
              # tokens = utterance.split()
              encoded_source = tokenizer.encode(utterance, max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt').squeeze()
              token_count = encoded_source.shape[0]
              # print(encoded_source.shape)
              repeated_utterance = [encoded_source for _ in range(token_count)]
              for pos in range(1, token_count):
                encoded_source = repeated_utterance[pos].clone()
                target_id = encoded_source[pos].item()
                if target_id == tokenizer.eos_token_id:
                  break

                encoded_source[pos] = tokenizer.mask_token_id
                decoded_target = ''.join(tokenizer.convert_ids_to_tokens([target_id])) + ' </s>'
                encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4, pad_to_max_length=True, truncation=True).squeeze() # should always be of size 1
                self.target.append(encoded_target)
                self.source.append(encoded_source)

              

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

  def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]#['input_ids'].squeeze()
        target_id = self.target[index]#['input_ids'].squeeze()
        # src_mask = self.source[index]['attention_mask'].squeeze()
        return { 'source_ids': source_ids,
                'target_id': target_id}

"""# Model"""

class T5MultiSPModel(pl.LightningModule):
  # def __init__(self, train_sampler=None, tokenizer= None, dataset=None, batch_size = 2):
  def __init__(self, hparams, task='denoise', test_flag='graphql', train_sampler=None, batch_size=2,temperature=1.0,top_k=50, top_p=1.0, num_beams=1 ):
    super(T5MultiSPModel, self).__init__()

    self.temperature = temperature
    self.top_k = top_k
    self.top_p = top_p
    self.num_beams = num_beams

    # self.lr=3e-5
    self.hparams = hparams

    self.task = task
    self.test_flag = test_flag
    self.train_sampler = train_sampler
    self.batch_size = batch_size
    # todo load from file if task is finetine. 
    if self.task == 'finetune':
      # have to change output_past to True manually
      self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
    else: 
      self.model = T5ForConditionalGeneration.from_pretrained('t5-base') # no output past? 

    self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    self.add_special_tokens()

  def forward(
    self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        lm_labels=lm_labels,
    )

  def add_special_tokens(self):
        # new special tokens
    special_tokens_dict = self.tokenizer.special_tokens_map # the issue could be here, might need to copy.
    special_tokens_dict['mask_token'] = '<mask>'
    special_tokens_dict['additional_special_tokens'] = ['<t>', '</t>', '<a>', '</a>']
    self.tokenizer.add_tokens(['{', '}', '<c>', '</c>'])
    self.tokenizer.add_special_tokens(special_tokens_dict)
    self.model.resize_token_embeddings(len(self.tokenizer))

  def _step(self, batch):
    if self.task == 'finetune':
      pad_token_id = self.tokenizer.pad_token_id
      source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
      lm_labels = y[:, :].clone()
      lm_labels[y[:, :] == pad_token_id] = -100
      outputs = self(source_ids, attention_mask=source_mask, lm_labels=lm_labels,)

      loss = outputs[0]

    else: 
      y = batch['target_id']
      lm_labels = y[:, :].clone()
      lm_labels[y[:, :] == self.tokenizer.pad_token_id] = -100
      loss = self(
          input_ids=batch["source_ids"],
          lm_labels=lm_labels
      )[0]


    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)
    return {"val_loss": loss}

  def validation_epoch_end(self, outputs):
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    tensorboard_logs = {"val_loss": avg_loss}
    return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs }
    

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
    if self.trainer.use_tpu:
      xm.optimizer_step(optimizer)
    else:
      optimizer.step()
    optimizer.zero_grad()
    self.lr_scheduler.step()


  def configure_optimizers(self):
    t_total = len(self.train_dataloader()) * self.trainer.max_epochs * self.trainer.train_percent_check
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )
    self.lr_scheduler = scheduler
    return [optimizer] #, [scheduler]

  def _generate_step(self, batch):
    generated_ids = self.model.generate(
        batch["source_ids"],
        attention_mask=batch["source_mask"],
        num_beams=self.num_beams,
        max_length=1000,
        temperature=self.temperature,
        top_k=self.top_k,
        top_p=self.top_p,
        # repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )

    preds = [
        self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]
    target = [
        self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for t in batch["target_ids"]
    ]
    return (preds, target)

  def test_step(self, batch, batch_idx):
    preds, target = self._generate_step(batch)
    loss = self._step(batch)
    if self.test_flag == 'graphql':
      accuracy = exact_match.exact_match_accuracy(preds,target)
      return {"test_loss": loss, "test_accuracy": torch.tensor(accuracy)}
    else: 
      return {"test_loss": loss, "preds": preds, "target": target }


  def test_epoch_end(self, outputs):
    avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    
    if self.test_flag == 'graphql':
      avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
      tensorboard_logs = {"test_loss": avg_loss, "test_acc": avg_acc}
      return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}

    else:
      output_test_predictions_file = os.path.join(os.getcwd(), "test_predictions.txt")
      with open(output_test_predictions_file, "w+") as p_writer:
          for output_batch in outputs:
              p_writer.writelines(s + "\n" for s in output_batch["preds"])
          p_writer.close()
      tensorboard_logs = {"test_loss": avg_loss}
      return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}

  def prepare_data(self):
    if self.task == 'finetune':
      self.train_dataset_g = TextToGraphQLDataset(self.tokenizer)
      self.val_dataset_g = TextToGraphQLDataset(self.tokenizer, type_path='dev.json')
      self.test_dataset_g = TextToGraphQLDataset(self.tokenizer, type_path='dev.json')

      self.train_dataset_s = SpiderDataset(self.tokenizer)
      self.val_dataset_s = SpiderDataset(self.tokenizer, type_path='dev.json')
      self.test_dataset_s = SpiderDataset(self.tokenizer, type_path='dev.json')

      self.train_dataset = ConcatDataset([self.train_dataset_g,self.train_dataset_s])
      self.val_dataset = ConcatDataset([self.val_dataset_g, self.val_dataset_s])
      # self.test_dataset = ConcatDataset([test_dataset_g, test_dataset_s])
      if self.test_flag == 'graphql':
        self.test_dataset = self.test_dataset_g
      else:
        self.test_dataset = self.test_dataset_s
      
    else:
      train_dataset_g = MaskGraphQLDataset(self.tokenizer)
      val_dataset_g = MaskGraphQLDataset(self.tokenizer, type_path='dev.json')

      train_dataset_s = CoSQLMaskDataset(self.tokenizer)
      val_dataset_s = CoSQLMaskDataset(self.tokenizer, type_path='cosql_dev.json')

      self.train_dataset = ConcatDataset([train_dataset_g, train_dataset_s])
      self.val_dataset = ConcatDataset([val_dataset_g,val_dataset_s])

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size)

"""# Pre-training"""

import argparse

hparams = argparse.Namespace(**{'lr': 0.0004365158322401656}) # for 3 epochs
system = T5MultiSPModel(hparams,batch_size=32)

from pytorch_lightning.callbacks import ModelCheckpoint
trainer = Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=1, train_percent_check=0.2)


trainer.fit(system)

"""Running the next two blocks probably uses memory unless I use without gradient.

"""

system.tokenizer.decode(system.train_dataset[0]['source_ids'].squeeze(), skip_special_tokens=False, clean_up_tokenization_spaces=False)

TXT = "query { faculty_aggregate { aggregate { <mask> } } } </s>"
input_ids = system.tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']

system.tokenizer.decode(system.model.generate(input_ids.cuda())[0])

"""# Finetune"""


system.hparams

system.task = 'finetune'
system.batch_size = 2 # because t5-base is smaller than bart.
system.hparams.lr=0.0005248074602497723 # same as 5e-4

system.prepare_data() # might not be needed. 


from pytorch_lightning.callbacks import ModelCheckpoint
trainer = Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=1, val_check_interval=0.5)

trainer.fit(system)

inputs = system.val_dataset[0]
system.tokenizer.decode(inputs['source_ids'])

generated_ids = system.model.generate(inputs['source_ids'].unsqueeze(0).cuda(), num_beams=5, repetition_penalty=1.0, max_length=56, early_stopping=True)

hyps = [system.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]

print(hyps)

system = system.load_from_checkpoint('finished.ckpt')
system.task='finetune'
trainer = Trainer(gpus=1, max_epochs=0, progress_bar_refresh_rate=1, val_check_interval=0.5)
trainer.fit(system)

"""# Test"""

system.num_beams = 3
system.test_flag = 'graphql'
system.prepare_data()
trainer.test()

system.num_beams = 3
system.test_flag = 'sql'
system.prepare_data()
trainer.test()

import nltk
nltk.download('punkt')

