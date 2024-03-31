# !pip install datasets rouge_score nltk
# !pip install accelerate -U
# !pip install transformers==4.27.0
# Transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM     # BERT Tokenizer and architecture
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments         # These will help us to fine-tune our model
from transformers import pipeline                                         # Pipeline
from transformers import DataCollatorForSeq2Seq                           # DataCollator to batch the data 
import string                                                              # PyTorch
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from datasets import load_metric, Dataset
import re


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test_text.csv')
val_data = pd.read_csv('validation.csv')


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


train_data = clean_df(train_data,['text', 'titles'])
test_data = clean_df(test_data,['text'])
val_data = clean_df(val_data,['text', 'titles'])


train_ds = Dataset.from_pandas(train_data)
test_ds = Dataset.from_pandas(test_data)
val_ds = Dataset.from_pandas(val_data)


checkpoint = 'facebook/bart-large-cnn' # Model
tokenizer = AutoTokenizer.from_pretrained(checkpoint) # Loading Tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to('cuda')


def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["titles"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Applying preprocess_function to the datasets
tokenized_train = train_ds.map(preprocess_function, batched=True,
                               remove_columns=['text', 'titles']) # Removing features
# Removing features
tokenized_val = val_ds.map(preprocess_function, batched=True,
                               remove_columns=['text', 'titles']) # Removing features


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


metric = load_metric('rouge')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred# Obtaining predictions and true labels
    
    # Decoding predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Obtaining the true labels tokens, while eliminating any possible masked token (i.e., label = -100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip(), language='french')) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip(), language='french')) for label in decoded_labels]
    
    
    # Computing rouge score
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()} # Extracting some results

    # Add mean-generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir = 'barthez',
    evaluation_strategy = "epoch",
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_loss',
    seed = 8,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=4,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    report_to="none"
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()