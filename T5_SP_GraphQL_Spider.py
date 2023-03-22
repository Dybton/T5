#!/usr/bin/env python
# coding: utf-8

# # Downloads, installs & imports

# SPEGQL Schemas and datasets download and unzip
# 

# In[26]:


# %%capture
# !gdown https://drive.google.com/uc?id=1H026ff-czIdLH3saJzbWUH8VKOW3j63X
# !unzip SPEGQL-dataset.zip


# Spider datasets download

# In[27]:


# %%capture
# !gdown https://drive.google.com/uc?id=14x6lsWqlu6gR-aYxa6cemslDN3qT3zxP
# !unzip cosql_dataset.zip


# In[28]:


# %%capture
# !gdown https://drive.google.com/uc?id=11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX
# !unzip spider.zip


# Installs

# In[52]:


# %%capture
# !pip install --upgrade jsonschema
# !pip3 install graphql-core --log ./logs.txt
# !pip3 install git+https://github.com/acarrera94/text-to-graphql-validation --no-deps --log ./logs.txt


# In[30]:


#!git clone https://github.com/huggingface/transformers
#!pip install transformers==4.15.0

import transformers
print("my version of transformers is " + transformers.__version__)

# !pip install torch==1.10.0
# !pip install pytorch-lightning 2.0.0


# In[31]:


import numpy as np
import pandas as pd

# # # from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import get_linear_schedule_with_warmup, AutoConfig 
from transformers import BartTokenizer,BartModel,BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers import BartConfig
from transformers import AutoTokenizer
from transformers import AdamW
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning import Trainer


import os
from os.path import basename
import re
from functools import reduce

# #from graphqlval import exact_match
import itertools

torch.manual_seed(0)

import json
from pathlib import Path
from os.path import basename
import glob
from functools import reduce

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)


# In[32]:


print("my version of pl is " + pl.__version__)

import transformers
import pytorch_lightning

print("my version of transformers is " + transformers.__version__)
print ("my version of pytorch is " + torch.__version__)
print("my version of pytorch_lightning is " +pytorch_lightning.__version__)


test_state = False
tensorflow_active = False

# # # Prepare GraphQL Dataset

# # In[33]:


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


# # In[34]:

if test_state:
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = TextToGraphQLDataset(tokenizer=tokenizer, type_path='train.json', block_size=102)

    length = dataset.__len__()
    item = dataset.__getitem__(0)
    print("TextToGraphQLDataset test done")


# So far:
# - the max length of a tokenized question is **49**
# - the max length of a tokenized query is **102**
# 
# 

# # In[35]:


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
          for example in data:

            utterance = example['query']
            encoded_source = tokenizer.encode(utterance, max_length=block_size, padding='max_length', truncation=True, return_tensors='pt').squeeze()
            token_count = encoded_source.shape[0]
            repeated_utterance = [encoded_source for _ in range(token_count)]
            for pos in range(1, token_count):
              encoded_source = repeated_utterance[pos].clone()
              target_id = encoded_source[pos].item()
              if target_id == tokenizer.eos_token_id:
                  break
              encoded_source[pos] = tokenizer.mask_token_id
              decoded_target = ''.join(tokenizer.convert_ids_to_tokens([target_id]))
              encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4, padding='max_length', truncation=True).squeeze()
              if encoded_target is not None and torch.numel(encoded_target) > 0:
                  self.target.append(encoded_target)
                  self.source.append(encoded_source)
              if torch.numel(encoded_target) > 0:
                  self.target.append(encoded_target)
                  self.source.append(encoded_source)


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

  def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]
        target_id = self.target[index]
        return { 'source_ids': source_ids,
                'target_id': target_id}


# # In[36]:

if test_state:
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    special_tokens_dict = tokenizer.special_tokens_map # the issue could be here, might need to copy.
    special_tokens_dict['mask_token'] = '<mask>'
    special_tokens_dict['additional_special_tokens'] = ['<t>', '</t>', '<a>', '</a>']
    tokenizer.add_tokens(['{', '}', '<c>', '</c>'])
    tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(len(tokenizer))
    print(tokenizer.mask_token)

    dataset = MaskGraphQLDataset(tokenizer=tokenizer, type_path='train.json', block_size=64)
    print("MaskGraphQLDataset test done")


# # # Prepare Spider Dataset

# # In[37]:


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
        # TODO open up tables.json
        # its a list of tables
        # group by db_id 
        # grab column name from column_names_original ( each column name is a list of two. and the 2nd index {1} is the column name )
        # grab table names from table_names (^ same as above )
        # concat both with the english question (table names + <c> + column names + <q> english question)
        # tokenize

        # Maybe try making making more structure 
        # in the concat by using primary_keys and foreign_keys 

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


            # group columns with tables. 

            db_with_question = 'translate English to SQL: ' + element['question'] + ' ' + tables_with_columns
            # question_with_schema = 'translate English to GraphQL: ' + element['question']  + ' ' + ' '.join(self.name_to_schema[element['schemaId']]) + ' </s>'

            tokenized_s = tokenizer.batch_encode_plus([db_with_question],max_length=1024, padding='max_length', truncation=True,return_tensors='pt')
            # what is the largest example size?
            # the alternative is to collate
            #might need to collate
            self.source.append(tokenized_s)

            tokenized_t = tokenizer.batch_encode_plus([element['query']],max_length=block_size, padding='max_length', truncation=True,return_tensors='pt')
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


# # In[38]:

if test_state:
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = SpiderDataset(tokenizer=tokenizer , type_path='train_spider.json', block_size=102)

    length = dataset.__len__()
    item = dataset.__getitem__(0)
    print("SpiderDataset test done")


# # In[39]:


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
              # repeat the squence for the amount of tokens. 
              # loop through those sequences and replace a different token in each one. 
              # the target will be that token. 
              utterance = interaction['query']
              # tokens = utterance.split()
              encoded_source = tokenizer.encode(utterance, max_length=block_size, padding='max_length', truncation=True, return_tensors='pt').squeeze()
              token_count = encoded_source.shape[0]
              # print(encoded_source.shape)
              repeated_utterance = [encoded_source for _ in range(token_count)]
              for pos in range(1, token_count):
                encoded_source = repeated_utterance[pos].clone()
                target_id = encoded_source[pos].item()
                if target_id == tokenizer.eos_token_id:
                  break
                # encoded_source[pos] = tokenizer.mask_token_id
                # self.target.append(target_id)
                # self.source.append(encoded_source)

                encoded_source[pos] = tokenizer.mask_token_id
                decoded_target = ''.join(tokenizer.convert_ids_to_tokens([target_id]))
                encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4, padding='max_length', truncation=True).squeeze() # should always be of size 1
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
                # 'source_mask': src_mask,
                # 'target_ids': target_ids,
                # 'target_ids_y': target_ids}


# In[40]:

if test_state:
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    special_tokens_dict = tokenizer.special_tokens_map # the issue could be here, might need to copy.
    special_tokens_dict['mask_token'] = '<mask>'
    special_tokens_dict['additional_special_tokens'] = ['<t>', '</t>', '<a>', '</a>']
    tokenizer.add_tokens(['{', '}', '<c>', '</c>'])
    tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(len(tokenizer))
    print(tokenizer.mask_token)

    dataset = CoSQLMaskDataset(tokenizer=tokenizer , type_path='cosql_train.json', block_size=64)

    length = dataset.__len__()
    item = dataset.__getitem__(0)
    print("CoSQLMaskDataset test done")


# # # Model

# # In[41]:


class T5MultiSPModel(pl.LightningModule):
  def __init__(self, hyperparams, task='denoise', test_flag='graphql', train_sampler=None, batch_size=2,temperature=1.0,top_k=50, top_p=1.0, num_beams=1 ):
    super(T5MultiSPModel, self).__init__()

    self.temperature = temperature
    self.top_k = top_k
    self.top_p = top_p
    self.num_beams = num_beams

    self.hyperparams = hyperparams

    self.task = task
    self.test_flag = test_flag
    self.train_sampler = train_sampler
    self.batch_size = batch_size
    if self.task == 'finetune':
      self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
    else: 
      self.model = T5ForConditionalGeneration.from_pretrained('t5-base') # no output past? 

    self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
    self.add_special_tokens()

  def forward(
    self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
    return self.model(
        input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        labels=labels,
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
      # y_ids = y[:, :-1].contiguous()
      labels = y[:, :].clone()
      labels[y[:, :] == pad_token_id] = -100
      # attention_mask is for ignore padding on source_ids 
      # labels need to have pad_token ignored manually by setting to -100
      # todo check the ignore token for forward
      # seems like decoder_input_ids can be removed. 
      outputs = self(source_ids, attention_mask=source_mask, labels=labels,)

      loss = outputs[0]

    else: 
      y = batch['target_id']
      labels = y[:, :].clone()
      labels[y[:, :] == self.tokenizer.pad_token_id] = -100
      loss = self(
          input_ids=batch["source_ids"],
          labels=labels
      )[0]


    return loss

  def training_step(self, batch, batch_idx):
    loss = self._step(batch)

    tensorboard_logs = {"train_loss": loss}
    return {"loss": loss, "log": tensorboard_logs}

  def validation_step(self, batch, batch_idx):
    loss = self._step(batch)

    print(f'Validation step called, batch_idx: {batch_idx}, loss: {loss.item()}')

    return {"val_loss": loss}


  def on_validation_epoch_end(self, outputs=None):
    if not outputs:
        print("Empty outputs list.")
        return
    print("outputs " + str(outputs))
    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    # if self.task == 'finetune':
    #   avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
    #   tensorboard_logs = {"val_loss": avg_loss, "avg_val_acc": avg_acc}
    #   return {"progress_bar": tensorboard_logs, "log": tensorboard_logs}
    # else:
    tensorboard_logs = {"val_loss": avg_loss}
    return {'progress_bar': tensorboard_logs, 'log': tensorboard_logs }
    

  # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
  #   if self.trainer:
  #     xm.optimizer_step(optimizer)
  #   else:
  #     optimizer.step()
  #   optimizer.zero_grad()
  #   self.lr_scheduler.step()


  def configure_optimizers(self):
    t_total = len(self.train_dataloader()) * self.trainer.max_epochs * self.trainer.limit_train_batches
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=self.hyperparams.lr, eps=1e-8)
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

  # def test_end(self, outputs):
  #   return self.validation_end(outputs)


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



# # In[42]:

# %load_ext tensorboard 

import sys
import subprocess

import sys
import subprocess

def launch_tensorboard_in_background(logdir):
    tensorboard_cmd = [sys.executable, "-m", "tensorboard", "--logdir", logdir]
    tensorboard_process = subprocess.Popen(tensorboard_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tensorboard_process

if tensorflow_active:
    if __name__ == "__main__":
        logdir = "lightning_logs/"

        tensorboard_process = launch_tensorboard_in_background(logdir)
        print(f"TensorBoard is running with logdir {logdir}")
        print("load tensorboard done")

        try:
            # Run your training code here, e.g., trainer.fit(system)

            # Keep the script running while TensorBoard is active
            while True:
                pass
        except KeyboardInterrupt:
            print("\nShutting down TensorBoard")
            tensorboard_process.terminate()
            tensorboard_process.wait()





# # # Pre-training

# # In[43]:


# # tokenizer = BartTokenizer.from_pretrained('bart-large')
# # dataset = ConvDataset(tokenizer)


# # In[44]:


import argparse


# # In[45]:


hyperparams = argparse.Namespace(**{'lr': 0.0004365158322401656}) # for 3 epochs


# # In[46]:


# # system = ConvBartSystem(dataset, train_sampler, batch_size=2)
system = T5MultiSPModel(hyperparams,batch_size=32)
print("We initialize the T5MultiSPModel(hyperparams,batch_size=32)")


# # In[47]:


from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
# Initialize the logger
logger = TensorBoardLogger("lightning_logs/")
# Pass the logger to the Trainer
trainer = pl.Trainer(logger=logger)

# # trainer = Trainer(num_tpu_cores=8,max_epochs=1)   
# # trainer = Trainer(max_epochs=1, limit_train_batches=0.1)
# # checkpoint_callback = ModelCheckpoint(
# #     filepath=os.getcwd()+'/checkpoint',
# #     verbose=True,
# #     monitor='val_loss',
# #     mode='min',
# #     prefix=''
# # )
# # trainer = Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=1)
# #trainer = Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=1, limit_train_batches=0.2)
trainer = Trainer(accelerator='cpu', max_epochs=1, log_every_n_steps=1, limit_train_batches=0.2)
# # trainer = Trainer(gpus=1, max_epochs=3, auto_lr_find=True, progress_bar_refresh_rate=1, limit_train_batches=0.2)\
# # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, limit_train_batches=0.2)
# # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, limit_train_batches=0.2,checkpoint_callback=checkpoint_callback)
# # trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)

# # In[48]:

# # Print the version of PyTorch Lightning
# print(pl.__version__)

# # In[50]:

# # import gc
# # gc.collect()
print("lets train this model!")
trainer.fit(system)

# # Running the next two blocks probably uses memory unless I use without gradient.
# # 

# # In[ ]:

# #system.tokenizer.decode(system.train_dataset[0]['source_ids'].squeeze(), skip_special_tokens=False, clean_up_tokenization_spaces=False)


# # In[ ]:


# # TXT = "query { faculty_aggregate { aggregate { <mask> } } } </s>"
# # input_ids = system.tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']

# # logits = system.model(input_ids)[0]


# # In[ ]:


# #system.tokenizer.decode(system.model.generate(input_ids.cuda())[0])


# # # Finetune

# # In[ ]:


# # len(system.val_dataloader().dataset)


# # In[ ]:


# # system = TextGraphQLModel.load_from_checkpoint('./_ckpt_epoch_0_v0.ckpt' )


# # In[ ]:


# #system.hyperparams


# # In[ ]:


# # system.task = 'finetune'
# # system.batch_size = 2 # because t5-base is smaller than bart.
# # # system.lr=3e-4 # -6 is original
# # # system.batch_size = 16
# # system.hyperparams.lr=0.0005248074602497723 # same as 5e-4
# # # system.hyperparams.lr=3e-4
# # # TODO collate to go back to 16
# # # system.model.config.output_past=True
# # # system.model.model.decoder.output_past=True
# # system.prepare_data() # might not be needed. 
# # # system.add_special_tokens()
# # # system.model.output_past = True


# # shouldn't we monitor: `avg_val_loss`?

# # In[ ]:


# # from pytorch_lightning.callbacks import ModelCheckpoint
# # # trainer = Trainer(num_tpu_cores=8,max_epochs=1)   
# # # trainer = Trainer(max_epochs=1, limit_train_batches=0.1)
# # # checkpoint_callback = ModelCheckpoint(
# # #     filepath=os.getcwd()+'/checkpoint_finetuning',
# # #     verbose=True,
# # #     monitor='val_loss',
# # #     mode='min',
# # #     prefix=''
# # # )

# # # trainer = Trainer(gpus=1,max_epochs=1, progress_bar_refresh_rate=1, limit_train_batches=0.2)
# # # trainer = Trainer(gpus=1, progress_bar_refresh_rate=1, val_check_interval=0.4)
# # trainer = Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=1, val_check_interval=0.5)
# # # trainer = Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=1, val_check_interval=0.5)
# # # trainer = Trainer(gpus=1,max_epochs=3, progress_bar_refresh_rate=1,checkpoint_callback=checkpoint_callback)
# # # trainer = Trainer(num_tpu_cores=8,max_epochs=1, progress_bar_refresh_rate=1)


# # In[ ]:


# # trainer.fit(system)


# # In[ ]:


# # inputs = system.val_dataset[0]
# # system.tokenizer.decode(inputs['source_ids'])

# # # system.tokenizer.decode(inputs['target_ids'])


# # In[ ]:


# # # inputs = system.tokenizer.batch_encode_plus([user_input], max_length=1024, return_tensors='pt')
# # # generated_ids = system.bart.generate(example['input_ids'].cuda(), attention_mask=example['attention_mask'].cuda(), num_beams=5, max_length=40,repetition_penalty=3.0)
# # # maybe i didn't need attention_mask? or the padding was breaking something.
# # # attention mask is only needed  
# # generated_ids = system.model.generate(inputs['source_ids'].unsqueeze(0).cuda(), num_beams=5, repetition_penalty=1.0, max_length=56, early_stopping=True)
# # # summary_text = system.tokenizer.decode(generated_ids[0])

# # hyps = [system.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]


# # In[ ]:


# # print(hyps)


# # improving the network: 
# # - can it handle <, > ? any other missing could check the dataset directy for unk
# # - try to increase the learning rate an order of magnitude.

# # In[ ]:


# # trainer.save_checkpoint('finished.ckpt')


# # In[ ]:


# # !zip -r finished_train.zip finished.ckpt


# # The cells below is used to load a checkpoint instead of training.

# # In[ ]:


# # from google.colab import drive
# # drive.mount('/content/gdrive')


# # In[ ]:


# # from google.colab import drive
# # drive.flush_and_unmount()


# # In[ ]:





# # In[ ]:


# # !cp ./drive/My\ Drive/ssh_files/finished_train.zip ./finished_train.zip


# # In[ ]:


# # !unzip ./finished_train.zip


# # In[ ]:


# # system = system.load_from_checkpoint('finished.ckpt')
# # system.task='finetune'
# # trainer = Trainer(gpus=1, max_epochs=0, progress_bar_refresh_rate=1, val_check_interval=0.5)
# # trainer.fit(system)


# # # Test

# # In[ ]:


# # system.num_beams = 3
# # system.test_flag = 'graphql'
# # system.prepare_data()
# # trainer.test()


# # In[ ]:


# # system.num_beams = 3
# # system.test_flag = 'sql'
# # system.prepare_data()
# # trainer.test()


# # In[ ]:


# # import nltk
# # nltk.download('punkt')


# # In[ ]:


# # !cd spider && git clone https://github.com/taoyds/spider


# # In[ ]:


# # !cd spider && python ./spider/evaluation.py --gold dev_gold.sql --pred ../test_predictions.txt --etype match --db ./database --table tables.json



# # # Inference server

# # In[ ]:


# # !pip install flask-ngrok


# # In[ ]:


# # pip install -U flask-cors


# # In[ ]:


# # from flask_ngrok import run_with_ngrok
# # from flask import Flask, jsonify, request, make_response
# # from flask_cors import CORS
# # app = Flask(__name__)
# # CORS(app)
# # run_with_ngrok(app)   #starts ngrok when the app is run

# # @app.route('/', methods=['GET', 'POST'])
# # @app.route("/predict", methods=['GET', 'POST'])
# # def predict():
# #   # if request.method == 'POST':
# #     # req_content = request.get_json()
# #     # print("request ", req_content)
# #   req_json = request.get_json()
# #   print(request)
# #   print(req_json)
# #   prompt = req_json['prompt']
# #   schemaId = req_json['schemaId']
# #   if system.train_dataset_g.name_to_schema[schemaId] is not None:
# #     input_string = system.train_dataset_g.get_question_with_schema(prompt, schemaId)
# #   elif system.dev_dataset.name_to_schema[schemaId] is not None:
# #     input_string = system.val_dataset_g.get_question_with_schema(prompt, schemaId)
# #   print(input_string)
  
# #   # val_inputs = system.val_dataset[0]
# #   # print(system.tokenizer.decode(val_inputs['source_ids'], skip_special_tokens=False))

# #   inputs = system.tokenizer.batch_encode_plus([input_string], max_length=1024, return_tensors='pt')['input_ids']

# #   print(inputs.shape)
# #   # print(val_inputs['source_ids'].shape)


# #   # generated_ids = system.model.generate(val_inputs['source_ids'].unsqueeze(0).cuda(), num_beams=1, repetition_penalty=1.0, max_length=1000, early_stopping=True)
# #   generated_ids = system.model.generate(inputs.cuda(), num_beams=3, repetition_penalty=1.0, max_length=1000, early_stopping=True)
# #   hyps = [system.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
# #   print(hyps)
# #   dict_res = { "prediction" : hyps[0]}
# #   print(dict_res)
# #   return jsonify(dict_res)


# # app.run()


# # In[ ]:



