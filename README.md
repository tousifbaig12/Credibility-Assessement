# Credibility-Assessement---Experience Mining

The objective of this project is to formulate a definition for credibility in general and most importantly how to define credibility from the students’ user group. The project also discusses the various factors identified by students to express whether the information is credible or not. Out of these, the project mainly concentrates on the factor “experience” to gauge the credibility. 

# Experimentation Code 

## Evaluating Experience - Using Machine Learning

!pip install pattern
### To evaluate sentence modality we use pattern library
### It gives score for a sentence between -1 to +1

import numpy as np
import pandas as pd
import nltk 
import re
import scipy.sparse as sp

from textblob import TextBlob
from pattern.en import modality
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('all')
### Downloading all the packages within NLTK

### Dataset

filePath = 'https://raw.githubusercontent.com/serenpa/Blog-Credibility-Corpus/main/blog_credibility_dataset.tsv'
### Convert accented characters to ASCII characters - encoding
df = pd.read_csv(filePath, sep='\t', encoding='latin1')
### Decoding to special characters
df['sentence_text'] = df.sentence_text.apply(lambda x:x.encode('latin1').decode('cp1252'))
### Removing empty spaces and special characters using regex
cleanParser=re.compile(r'(\w+)')
### Replacing multiple spaces or special characters to single space in a single sentence plus lowering characters
df['cleanSentence'] = df.sentence_text.apply(lambda x: ' '.join([i.lower() for i in cleanParser.findall(x.strip())]))
df.head()

print('#Unique document Ids',df.document_id.nunique())

df.Experience.value_counts()

### Features

def getSubjectivity(sentence):
  ''' Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information'''
  textBlobObj = TextBlob(sentence)
  ### rounding subjectivity score to 4 decimals
  return round(textBlobObj.subjectivity, 4)

### Feature 1-4 - Sentiment Intensity Compound, Positive, Negative, Neutral
'''The Compound score is a metric that calculates the sum of all the
 lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive).
 positive sentiment : (compound score >= 0.05) 
 neutral sentiment : (compound score > -0.05) and (compound score < 0.05) 
 negative sentiment : (compound score <= -0.05) '''

def getSentimentIntensity(sentence):
  vad = SentimentIntensityAnalyzer()
  scores = vad.polarity_scores(sentence)
  return scores.get('compound'), scores.get('neg'), scores.get('neu'), scores.get('pos')

### Feature 5 - Retrieving POS  tags from the sentence
def getPOSTags(sentence):
  from collections import Counter
  words = nltk.word_tokenize(sentence)
  pos_tags = nltk.pos_tag(words)
  return pos_tags, Counter(tag for word, tag in pos_tags).items()

### Feature 6 - Named entity recognition
def identifyNER(ptags):
  ners = nltk.ne_chunk(ptags, binary=True)
  words = [i[0] for i in ners if isinstance(i, nltk.Tree)]
  return ' '.join(i[0] for i in words) if words else np.nan

### Feaature 7 - Count of I
def countIs(sentence):
  return len(re.findall(r'i | i ', sentence.lower()))

### Feature 8 - Modality score for given sentence
'''
Modality is a semantic notion that is related to speaker’s opinion and belief 
about the event’s believability. Modality in English can be achieved by modal verbs (will/would)
'''
def getModality(sentence):
  try:
    return modality(sentence)
  except RuntimeError as e:
    return 0

### subjectivity
df['subjectivity'] = df.apply(lambda x: getSubjectivity(x.cleanSentence), axis='columns')

### Feature 9 - Word count
df['wordCount'] = df.cleanSentence.apply(lambda x: len(x.split(' ')))

### POS tags
df[['pos_tags', 'pos_Density']] = df.apply(lambda x: getPOSTags(x.cleanSentence), axis='columns', result_type='expand')

### Sentiment intensity
df[['compoundIntensity', 'negativeIntensity', 'neutralIntensity', 'positiveIntensity']] = df.apply(lambda x: \
                                          getSentimentIntensity(x.cleanSentence), axis='columns', result_type='expand')

### NER recognition
df['ner_Density'] = df.pos_tags.apply(identifyNER)

### Count of I
df['i_count'] = df.cleanSentence.apply(countIs)

df['ner_count'] = df.ner_Density.apply(lambda x: 0 if isinstance(x, float) else len(x.split(' ')))

### Feature 10 - Char count
df['char_count'] = df.cleanSentence.apply(lambda x:len(x))
### Feature 11 - Average word length
df['avg_word_len'] = df.cleanSentence.apply(lambda x: np.mean([len(i) for i in x.split(' ')]))
df['modality'] = df.cleanSentence.apply(getModality)

### Features 12 - POS count
df['NN_Count'] = df.pos_Density.apply(lambda x: dict(x).get('NN',0))
df['IN_Count'] = df.pos_Density.apply(lambda x: dict(x).get('IN',0))
df['DT_Count'] = df.pos_Density.apply(lambda x: dict(x).get('DT',0))
df['JJ_Count'] = df.pos_Density.apply(lambda x: dict(x).get('JJ',0))
df['NNS_Count'] = df.pos_Density.apply(lambda x: dict(x).get('NNS',0))
df['PRP_Count'] = df.pos_Density.apply(lambda x: dict(x).get('PRP',0))
df['RB_Count'] = df.pos_Density.apply(lambda x: dict(x).get('RB',0))
df['VB_Count'] = df.pos_Density.apply(lambda x: dict(x).get('VB',0))
df['VBP_Count'] = df.pos_Density.apply(lambda x: dict(x).get('VBP',0))
df['TO_Count'] = df.pos_Density.apply(lambda x: dict(x).get('TO',0))
df['VBZ_Count'] = df.pos_Density.apply(lambda x: dict(x).get('VBZ',0))
df['CC_Count'] = df.pos_Density.apply(lambda x: dict(x).get('CC',0))
df['VBD_Count'] = df.pos_Density.apply(lambda x: dict(x).get('VBD',0))
df['VBG_Count'] = df.pos_Density.apply(lambda x: dict(x).get('VBG',0))
df['VBN_Count'] = df.pos_Density.apply(lambda x: dict(x).get('VBN',0))
df['CD_Count'] = df.pos_Density.apply(lambda x: dict(x).get('CD',0))
df.head()

### Importing Models

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

## Train Test Split

# Custom feature list
customFeatures = ['wordCount', 'subjectivity','compoundIntensity','negativeIntensity',
       'neutralIntensity', 'positiveIntensity','i_count',
       'ner_count', 'char_count', 'avg_word_len','modality',
       'NN_Count', 'IN_Count', 'DT_Count', 'JJ_Count', 'NNS_Count', 'PRP_Count',
       'RB_Count', 'VB_Count', 'VBP_Count', 'TO_Count', 'VBZ_Count',
       'CC_Count', 'VBD_Count','VBG_Count', 'VBN_Count', 'CD_Count']


splitColumns = ['sentence_text'] + customFeatures
samSize = len(df[df.Experience==1])
df_sampled = pd.concat([df[df.Experience==0].sample(samSize, random_state=150), df[df.Experience==1]])

# shuffling the combined dataset of balanced data i-e experience and non experience
df_sampled = df_sampled.sample(frac=1, random_state=50).reset_index(drop=True)
scaler = MinMaxScaler()
df_exp = pd.DataFrame(scaler.fit_transform(df_sampled[customFeatures]), columns=customFeatures)
df_exp['sentence_text']=df_sampled['sentence_text']
df_exp['Experience']=df_sampled['Experience']

# train test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(df_exp[splitColumns], df_exp.Experience, \
                                                    test_size=0.2, random_state=10, stratify=df_exp.Experience)


# Creating train & test for 3 different combinations TF-IDF, only Custom features, TF-IDF + Custom features

vecModel = TfidfVectorizer(analyzer='word', stop_words='english')
train = vecModel.fit_transform(X_train.sentence_text)
test = vecModel.transform(X_test.sentence_text)

# Stacking TF-IDF sparse matrix with custom features
train_2 = sp.hstack([train, X_train[customFeatures]])
test_2 = sp.hstack([test, X_test[customFeatures]])

#Only custom features
train_3 = X_train[customFeatures]
test_3 =  X_test[customFeatures]

# Models

models = {'SVM':SVC(C=1, kernel='linear', random_state=10),\
          'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_split=5),\
          'Decision Tree': DecisionTreeClassifier(random_state=10), \
          'KNN': KNeighborsClassifier(n_neighbors=2),\
          'Naive Bayes': MultinomialNB()}

## Only with TFIDF

rslt = list()
for name, m in models.items():
  m.fit(train, y_train)
  pred = m.predict(test)
  # Gives clsasification report which gives Accuracy, precision, recall and f1
  report = classification_report(y_test, pred, output_dict=True)
  
  # To format in the dataframe for experience records, we use get method
  rslt.append({'Model': name,\
               'Accuracy': round(accuracy_score(y_test, pred)*100, 2),\
               'Precision': round(report.get('1').get('precision')*100, 2),\
               'Recall': round(report.get('1').get('recall')*100, 2),\
               'F1': round(report.get('1').get('f1-score')*100, 2)})

df_classifier = pd.DataFrame(rslt)
df_classifier.sort_values(by='Accuracy', ascending=False, inplace=True)
df_classifier.reset_index(drop=True, inplace=True)
df_classifier

## Only Custom Features

rslt = list()
for name, m in models.items():
  
  m.fit(train_3, y_train)
  pred = m.predict(test_3)
  report = classification_report(y_test, pred, output_dict=True)
  
  rslt.append({'Model': name,\
               'Accuracy': round(accuracy_score(y_test, pred)*100, 2),\
               'Precision': round(report.get('1').get('precision')*100, 2),\
               'Recall': round(report.get('1').get('recall')*100, 2),\
               'F1': round(report.get('1').get('f1-score')*100, 2)})
df_classifier = pd.DataFrame(rslt)
df_classifier.sort_values(by='Accuracy', ascending=False, inplace=True)
df_classifier.reset_index(drop=True, inplace=True)
df_classifier

## TFIDF + Custom features

rslt = list()
for name, m in models.items():
  m.fit(train_2, y_train)
  pred = m.predict(test_2)
  report = classification_report(y_test, pred, output_dict=True)
  
  rslt.append({'Model': name,\
               'Accuracy': round(accuracy_score(y_test, pred)*100, 2),\
               'Precision': round(report.get('1').get('precision')*100, 2),\
               'Recall': round(report.get('1').get('recall')*100, 2),\
               'F1': round(report.get('1').get('f1-score')*100, 2)})
df_classifier = pd.DataFrame(rslt)
df_classifier.sort_values(by='Accuracy', ascending=False, inplace=True)
df_classifier.reset_index(drop=True, inplace=True)
df_classifier



# Evaluating Experience - Using Deep Learning Algorithm BERT

# Pre-Trained BERT 12-Layer Model





# Loss function - Cross Entropy loss
# Optimizer - AdamW

!pip install transformers
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AdamW
from tensorflow.keras.preprocessing.sequence import pad_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.get_device_name(0)

filePath = 'https://raw.githubusercontent.com/serenpa/Blog-Credibility-Corpus/main/blog_credibility_dataset.tsv'
df = pd.read_csv(filePath, sep='\t', encoding='latin1')
df['sentence_text'] = df.sentence_text.apply(lambda x:x.encode('latin1').decode('cp1252'))
samSize = len(df[df.Experience==1])
df_sampled = pd.concat([df[df.Experience==0].sample(samSize, random_state=50), df[df.Experience==1]])
df_sampled = df_sampled.sample(frac=1,random_state=50).reset_index(drop=True)
df_sampled.Experience.value_counts()
# Creating a 'list' of sentences using data frame
sentences = df_sampled.sentence_text.values

sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df_sampled.Experience.values
unique, counts = np.unique(labels, return_counts=True)
dict(zip(unique, counts))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print(sentences[0])
print ("Tokenize the first sentence:")
print (tokenized_texts[0])

BERT requires specifically formatted inputs. For each tokenized input sentence, we need to create:

- **input ids**: a sequence of integers identifying each input token to its index number in the BERT tokenizer vocabulary
- **segment mask**: (optional) a sequence of 1s and 0s used to identify whether the input is one sentence or two sentences long. For one sentence inputs, this is simply a sequence of 0s. For two sentence inputs, there is a 0 for each token of the first sentence, followed by a 1 for each token of the second sentence
- **attention mask**: (optional) a sequence of 1s and 0s, with 1s for all input tokens and 0s for all padding tokens
- **labels**: a single value of 1 or 0. In our task 1 means "grammatical" and 0 means "ungrammatical"

Although we can have variable length input sentences, BERT does requires our input arrays to be the same size. We address this by first choosing a maximum sentence length, and then padding and truncating our inputs until every input sequence is of the same length.</p><p>To "pad" our inputs in this context means that if a sentence is shorter than the maximum sentence length, we simply add 0s to the end of the sequence until it is the maximum sentence length.</p><p>If a sentence is longer than the maximum sentence length, then we simply truncate the end of the sequence, discarding anything that does not fit into our maximum sentence length.</p><p>We pad and truncate our sequences so that they all become of length MAX_LEN ("post" indicates that we want to pad and truncate at the end of the sequence, as opposed to the beginning) pad_sequences is a utility function that we're borrowing from Keras. It simply handles the truncating and padding of Python lists.

# BERT tokenizer to convert tokens to their respectively index number as per BERT vocabulary
MAX_LEN = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
print(len(input_ids[0]))
#----
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
print(len(input_ids[0]))

# Attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=50, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=50, test_size=0.2)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

OK, let's load BERT! There are a few different pre-trained BERT models available. "bert-base-uncased" means the version that has only lowercase letters ("uncased") and is the smaller version of the two ("base" vs "large").

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

Now that we have our model loaded we need to grab the training hyperparameters from within the stored model.

For the purposes of fine-tuning, the authors recommend the following hyperparameter ranges:
- Batch size: 16
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 4, 5, 6

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

Below is our training loop. There's a lot going on, but fundamentally for each pass in our loop we have a trianing phase and a validation phase. At each pass we need to:

Training loop:
- Tell the model to compute gradients by setting the model in train mode
- Unpack our data inputs and labels
- Load data onto the GPU for acceleration
- Clear out the gradients calculated in the previous pass. In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out
- Forward pass (feed input data through the network)
- Backward pass (backpropagation)
- Tell the network to update parameters with optimizer.step()
- Track variables for monitoring progress

Evalution loop:
- Tell the model not to compute gradients by setting th emodel in evaluation mode
- Unpack our data inputs and labels
- Load data onto the GPU for acceleration
- Forward pass (feed input data through the network)
- Compute loss on our validation data and track variables for monitoring progress

t = [] 
trueLabels = []
predLabels = []
# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 5
loss=torch.nn.CrossEntropyLoss()
# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  # Training
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # batch = tuple(t for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    loss = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)[0]

    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1
  print("Train loss: {}".format(tr_loss/nb_tr_steps))
  # Validation
  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    #batch = tuple(t for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids.long(), token_type_ids=None, attention_mask=b_input_mask)[0]
    
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    trueLabels.extend(label_ids)
    predLabels.extend(np.argmax(logits, axis=1).flatten())

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

from sklearn.metrics import classification_report
print(classification_report(trueLabels, predLabels))
