import transformers
print("my version of transformers is " + transformers.__version__)


# In[31]:
import numpy as np
import pandas as pd
import torch
print("torch.cuda.is_available()")
print(torch.cuda.is_available())
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


import os
import socket
from os.path import basename
import re
from functools import reduce

import itertools

torch.manual_seed(0)

import json
from pathlib import Path
from os.path import basename
import glob
from functools import reduce


print("my version of pl is " + pl.__version__)

import transformers
import pytorch_lightning

print("my version of transformers is " + transformers.__version__)
print ("my version of pytorch is " + torch.__version__)
print("my version of pytorch_lightning is " +pytorch_lightning.__version__)


test_state = False
tensorflow_active = False

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
            question_with_schema = 'translate English to GraphQL: ' + element['question']  + ' ' + ' '.join(self.name_to_schema[element['schemaId']]) + ' </s>'
            tokenized_s = tokenizer.encode_plus(question_with_schema, max_length=1024, pad_to_max_length=True, truncation=True, return_tensors='pt')
            self.source.append(tokenized_s)

            tokenized_t = tokenizer.encode_plus(element['query'] + ' </s>', max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt')
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


if test_state:
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = TextToGraphQLDataset(tokenizer=tokenizer, type_path='train.json', block_size=102)

    length = dataset.__len__()
    item = dataset.__getitem__(0)
    print("TextToGraphQLDataset test done")


class MaskGraphQLDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, tokenizer, type_path='train.json', block_size=64):
        'Initialization'
        super(MaskGraphQLDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        path = './SPEGQL-dataset/dataset/' + type_path
        with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
          for example in data:

            utterance = example['query']
            encoded_source = tokenizer.encode(utterance + ' </s>', max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt').squeeze()
            token_count = encoded_source.shape[0]
            repeated_utterance = [encoded_source for _ in range(token_count)]
            for pos in range(1, token_count):
              encoded_source = repeated_utterance[pos].clone()
              target_id = encoded_source[pos].item()
              if target_id == tokenizer.eos_token_id:
                  break
              encoded_source[pos] = tokenizer.mask_token_id
              decoded_target = ''.join(tokenizer.convert_ids_to_tokens([target_id])) + ' </s>'
              encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4, pad_to_max_length=True, truncation=True).squeeze()
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

          grouped_dbs = {}
          for db in databases:
            grouped_dbs[db['db_id']] = db

          for element in data:
            db = grouped_dbs[element['db_id']]
            db_tables = db['table_names_original']
            tables_with_columns = ''
            for table_id, group in itertools.groupby(db['column_names_original'], lambda x: x[0]):
              if table_id == -1:
                continue

              columns_names = " ".join([column_name[1] for column_name in group ])
              tables_with_columns += '<t> ' + db_tables[table_id] + ' <c> ' + columns_names + ' </c> ' + '</t> '

            db_with_question = 'translate English to SQL: ' + element['question'] + ' ' + tables_with_columns

            tokenized_s = tokenizer.encode_plus(db_with_question, max_length=1024, pad_to_max_length=True, truncation=True, return_tensors='pt')
            self.source.append(tokenized_s)

            tokenized_t = tokenizer.encode_plus(element['query'], max_length=block_size, pad_to_max_length=True, truncation=True, return_tensors='pt')
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

if test_state:
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = SpiderDataset(tokenizer=tokenizer , type_path='train_spider.json', block_size=102)

    length = dataset.__len__()
    item = dataset.__getitem__(0)
    print("SpiderDataset test done")

class CoSQLMaskDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, tokenizer, type_path='cosql_train.json', block_size=64):
        'Initialization'
        super(CoSQLMaskDataset, ).__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []
        path = './cosql_dataset/sql_state_tracking/' + type_path
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for element in data:
                for interaction in element['interaction']:
                    utterance = interaction['query']
                    encoded_source = tokenizer.encode(utterance, max_length=block_size, padding='max_length', truncation=True, return_tensors='pt').squeeze()
                    token_count = encoded_source.shape[0]
                    repeated_utterance = [encoded_source for _ in range(token_count)]
                    for pos in range(1, token_count):
                        encoded_source = repeated_utterance[pos].clone()
                        target_id = encoded_source[pos].item()
                        if target_id == tokenizer.eos_token_id:
                            break

                        encoded_source[pos] = tokenizer.mask_token_id
                        decoded_target = ''.join(tokenizer.convert_ids_to_tokens([target_id])) + tokenizer.eos_token
                        encoded_target = tokenizer.encode(decoded_target, return_tensors='pt', max_length=4, padding='max_length', truncation=True).squeeze()
                        self.target.append(encoded_target)
                        self.source.append(encoded_source)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.source)

    def __getitem__(self, index):
        'Generates one sample of data'
        source_ids = self.source[index]
        target_id = self.target[index]
        return {'source_ids': source_ids,
                'target_id': target_id}


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
    

  def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
      if on_tpu or self.trainer.accelerator_connector.use_tpu:
          xm.optimizer_step(optimizer)
      else:
          optimizer.step()
      optimizer.zero_grad()
      self.lr_scheduler.step()



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

  def exact_match_accuracy(self, preds, target):
    total = len(preds)
    correct = sum([1 for pred, tgt in zip(preds, target) if pred == tgt])
    return correct / total

  def test_step(self, batch, batch_idx):
    preds, target = self._generate_step(batch)
    loss = self._step(batch)
    if self.test_flag == 'graphql':
      accuracy = self.exact_match_accuracy(preds, target)
      return {"test_loss": loss, "test_accuracy": torch.tensor(accuracy)}
    else:
      return {"test_loss": loss, "preds": preds, "target": target }

  # def test_end(self, outputs):
  #   return self.validation_end(outputs)

  def test_epoch_end(self, outputs):
    avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    self.log("avg_test_loss", avg_loss, prog_bar=True)

    if self.test_flag == 'graphql':
        avg_accuracy = torch.stack([x["test_accuracy"] for x in outputs if "test_accuracy" in x]).mean()
        self.log("avg_test_accuracy", avg_accuracy, prog_bar=True)
        return {"avg_test_loss": avg_loss, "avg_test_accuracy": avg_accuracy}

    else:
        output_test_predictions_file = os.path.join(os.getcwd(), "test_predictions.txt")
        with open(output_test_predictions_file, "w+") as p_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
            p_writer.close()
        return {"avg_test_loss": avg_loss}

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

import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

hyperparams = argparse.Namespace(**{'lr': 0.0004365158322401656}) # for 3 epochs

system = T5MultiSPModel(hyperparams, batch_size=32)
print("We initialize the T5MultiSPModel(hyperparams,batch_size=32)")

# Initialize the logger
logger = TensorBoardLogger("lightning_logs/")

## Initial Training

script_dir = os.path.dirname(os.path.realpath(__file__))

# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=script_dir,
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

# Pass the logger and checkpoint_callback to the Trainer
trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], accelerator='gpu', max_epochs=1, log_every_n_steps=1, limit_train_batches=0.2, gpus=1)

# Train the model
trainer.fit(system)

system.prepare_data() # might not be needed.

# Load the best initial training model for testing
best_initial_training_model = T5MultiSPModel.load_from_checkpoint(checkpoint_callback.best_model_path)

### Testing the model
system.tokenizer.decode(best_initial_training_model.train_dataset[0]['source_ids'].squeeze(), skip_special_tokens=False, clean_up_tokenization_spaces=False)

TXT = "query { faculty_aggregate { aggregate { <mask> } } } </s>"
input_ids = best_initial_training_model.tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
best_initial_training_model.model.cuda()

best_initial_training_model.tokenizer.decode(system.model.generate(input_ids.cuda())[0])

# Fine Tuning
system.task = 'finetune'
system.batch_size = 2 # because t5-base is smaller than bart.
system.hyperparams.lr=0.0005248074602497723 # same as 5e-4
system.prepare_data() # might not be needed.

# Create another ModelCheckpoint callback for fine-tuning
fine_tuning_checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=script_dir,
    filename='fine_tuned_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
)

# Pass the new checkpoint_callback to the Trainer
trainer = Trainer(gpus=1, max_epochs=6, progress_bar_refresh_rate=1, val_check_interval=0.5, callbacks=[fine_tuning_checkpoint_callback])

# Fine-tune the model
trainer.fit(system)

# Load the best fine-tuned model for testing
best_fine_tuned_model = T5MultiSPModel.load_from_checkpoint(fine_tuning_checkpoint_callback.best_model_path)

inputs = best_fine_tuned_model.val_dataset[0]
best_fine_tuned_model.tokenizer.decode(inputs['source_ids'])


best_fine_tuned_model.model = best_fine_tuned_model.model.cuda()
generated_ids = best_fine_tuned_model.model.generate(inputs['source_ids'].unsqueeze(0).cuda(), num_beams=5, repetition_penalty=1.0, max_length=56, early_stopping=True)

hyps = [best_fine_tuned_model.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]

print("hyps")
print(hyps)


## Testing

best_fine_tuned_model.num_beams = 3
best_fine_tuned_model.test_flag = 'graphql'
best_fine_tuned_model.prepare_data()
trainer.test(model=best_fine_tuned_model)
