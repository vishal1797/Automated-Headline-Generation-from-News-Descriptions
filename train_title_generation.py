#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer


# In[ ]:


pip install openai


# In[ ]:


pip install torch


# In[ ]:


pip install transformers


# # Data Preprocessing

# In[ ]:


data=pd.read_csv('/content/drive/MyDrive/CloudSEK/Headline creation /train.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


train_data=data.dropna()
train_data.isnull().sum()


# In[ ]:


train_data['text'] = train_data.apply(lambda row: f"Summary: {row['Summary']} Headline: {row['Headline']}", axis=1)

print(f"Number of rows after preprocessing: {len(train_data)}")


# In[ ]:


test_data = pd.read_csv('/content/drive/MyDrive/CloudSEK/Headline creation /test.csv')


# In[ ]:


test_data.head()


# In[ ]:


test_data.isnull().sum()
test_data=test_data.dropna()


# In[ ]:


test_data['text'] = test_data.apply(lambda row: f"Summary: {row['Summary']} Headline: {row['Headline']}", axis=1)
print(f"Number of rows after preprocessing: {len(test_data)}")


# # Tokenization

# In[ ]:


pip install datasets


# In[ ]:


from datasets import Dataset
from transformers import GPT2Tokenizer


# In[ ]:


# Convert to Hugging Face dataset

dataset = Dataset.from_pandas(train_data[['text']])
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
val_dataset = dataset['test']


# In[ ]:


# tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '<pad>'})

def tokenize_function(examples):

    encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    encoding['labels'] = encoding['input_ids'].copy()
    return encoding

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)


train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])


# # Fine Tune GPT2 Model

# In[ ]:


pip install accelerate -U


# In[ ]:


pip show accelerate


# In[ ]:


from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import accelerate
import torch


# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


trainer.train()


# # Result generation and model saving

# In[ ]:


def generate_headline(summary):
    prompt = f"Summary: {summary} Headline:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


summary = "The stock market saw a significant increase today as major tech companies reported record earnings for the quarter."
headline = generate_headline(summary)
print("Generated Headline:", headline)


# In[1]:


model.save_pretrained('./fine_tuned_gpt2')
tokenizer.save_pretrained('./fine_tuned_gpt2')

