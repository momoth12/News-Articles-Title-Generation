from transformers import pipeline   
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import string                                                              
import pandas as pd
import re
import os
from tqdm import tqdm

test_data = pd.read_csv('test_text.csv')

def clean_text(text):
        text = text.lower()
        text = re.sub('^.*?- ', '', text)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text

def clean_text_source(text):
        text = text.lower()
        text = re.sub('^.*?- ', '', text)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return 'summarize: ' + text

def clean_df(df, cols):
    for col in cols:
        if col == 'text':
            df[col] = df[col].fillna('').apply(clean_text_source)
        else:
            df[col] = df[col].fillna('').apply(clean_text)
    return df

test_data = clean_df(test_data,['text'])

dirs = os.listdir('barthez')
tokenizer_kwargs = {'max_length':1024, 'truncation':True}

i = 0
for dir in dirs:
    if i == 0:
        i += 1
        continue
    test_dataset = Dataset.from_pandas(test_data)
    pipeline_sum = pipeline('summarization', model=f'barthez/{dir}', device=0, batch_size=4)
    for i, title in tqdm(enumerate(pipeline_sum(KeyDataset(test_dataset, "text"), **tokenizer_kwargs))):
        test_data.loc[i, 'titles'] = title[0]['summary_text']
    test_data.drop('text', axis=1, inplace=True)
    test_data.to_csv(f'{dir}_test.csv', index=False)
    print(f'{dir} done')