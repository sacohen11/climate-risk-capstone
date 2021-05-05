#################################################################
# Title: Climate Identification Models
# Author: Sam Cohen

# Notes:
# These functions encompass loading training data, testing various models, and
# making predictions on the climate-related sentences from the 10-K file.

# Based on Ask BERT: How Regulatory Disclosure of Transition and Physical
# Climate Risks affects the CDS Term Structure(Nov. 2020, Kolbel)
##################################################################

# Packages
import pandas as pd
import numpy as np
import os
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizer, XLNetTokenizer, XLNetModel, OpenAIGPTTokenizer, OpenAIGPTModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn import metrics
import concurrent.futures as cf
import multiprocessing
import time
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from textwrap import wrap
import functools

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# Set Global Parameters
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss = nn.CrossEntropyLoss().to(device)
N_EPOCHS = 4
BATCH_SIZE = 16


def load_training_data():
    """
    This function builds the dataset to fine-tune the BERT model.
    It consists of sentences from TCFD Good Practices Handbook (https://www.cdsb.net/sites/default/files/tcfd_good_practice_handbook_web_a4.pdf)
    and random sentences form both CNN and other websites.

    :return: x_train, y_train, x_test, y_test, x_val, y_val
    """
    print("Load data and create dataset...", end='')

    # Load TCFD Sentences
    urls = pd.read_excel("training_sentences.xlsx", sheet_name="Sheet1", index_col=0)
    sents = urls[urls['whole']==0]
    sents = sents.reset_index()
    climate_sents = np.array(sents["sentences"]).flatten()
    target_climate = np.ones(len(climate_sents))


    # Load Random sentences from CNN and other websites
    # Using array created from the script "random_sentences.py"
    random = np.load("random.npy")
    target_random = np.zeros(len(random))

    # Load Random sentences from CNN and other websites
    # Using array created from the script "random_sentences.py"
    ten_k_supplements = pd.read_excel("training_sentences.xlsx", sheet_name="Sheet4", index_col=0)
    sents = np.array(ten_k_supplements["ten_k_sents"]).flatten()
    labels = np.array(ten_k_supplements["classifications"]).flatten()

    # Append datasets (both target and sentences) together
    x = np.hstack((climate_sents, random, sents))
    y = np.hstack((target_climate, target_random, labels))

    # Train, test, split into training, testing, and validation sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=RANDOM_SEED, shuffle=True, test_size=.25)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=RANDOM_SEED, shuffle=True, test_size=.5)

    print('done')
    print("The number of climate related sentences is:", len(climate_sents))
    print("The number of non-climate related sentences is:", len(random))
    print("The number of hand labeled 10-K sentences is:", len(sents))

    return x_train, y_train, x_test, y_test, x_val, y_val

class ClimateSentencesDataset(Dataset):

    """
    This class instantiates a Climate Sentences Dataset.
    This dataset class holds the main features related to climate change sentences.
    It also encodes a sentence into tokens to be used in the BERT model.
    """

    def __init__(self, sentences, targets, tokenizer):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        target = self.targets[item]
        tokenizer = self.tokenizer

        encoding = tokenizer.encode_plus(
            sentence,
            max_length=150,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            truncation=True,
            #padding='longest',
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',  # Return PyTorch tensors
        )
        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(sentences, targets, tokenizer, batch_size):
    """
    The data loader function creates a Climate Sentence Dataset from input arrays.
    :param sentences: 10-K or training sentences
    :param targets: Binary encoding for whether the sentence is climate-related (1) or not (0). If this is the prediction data, this parameter does not matter.
    :param tokenizer: Tokenized sentences
    :param batch_size: batch size of the data loader
    :return: a data loader to input into the modeling pipeline
    """
    ds = ClimateSentencesDataset(
        sentences=sentences,
        targets=targets,
        tokenizer=tokenizer)

    return DataLoader(
        ds,
        batch_size=batch_size)

def tokenize_and_create_loaders(x_train, y_train, x_test, y_test, x_val, y_val, PRE_TRAINED_MODEL_NAME):
    """

    :param x_train: training sentences
    :param y_train: training targets
    :param x_test: testing sentences
    :param y_test: testing targets
    :param x_val: validation sentences
    :param y_val: validation targets
    :param PRE_TRAINED_MODEL_NAME: type of model
    :return: a data loader for the training, testing, and validation datasets
    """

    print("Tokenize and create data loaders...", end='')

    # This bit of code accomodates all 5 model types
    if PRE_TRAINED_MODEL_NAME[0:4] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    elif PRE_TRAINED_MODEL_NAME[0:7] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    elif PRE_TRAINED_MODEL_NAME[0:5] == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    elif PRE_TRAINED_MODEL_NAME[0:6] == 'openai':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # create data loaders
    train_data_loader = create_data_loader(x_train, y_train, tokenizer, BATCH_SIZE)
    test_data_loader = create_data_loader(x_test, y_test, tokenizer, BATCH_SIZE)
    val_data_loader = create_data_loader(x_val, y_val, tokenizer, BATCH_SIZE)

    print('done')

    return train_data_loader, test_data_loader, val_data_loader

class ClimateIdentifier(nn.Module):
    """
    Instantiates a Climate Identifier model.
    This model has one fine-tuning layer and ends in a softmax function to identify the target with the maximum probability of being classified.
    """
    def __init__(self, PRE_TRAINED_MODEL_NAME):
        super(ClimateIdentifier, self).__init__()
        if PRE_TRAINED_MODEL_NAME[0:4] == 'bert':
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        elif PRE_TRAINED_MODEL_NAME[0:7] == 'roberta':
            self.bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        elif PRE_TRAINED_MODEL_NAME[0:5] == 'xlnet':
            self.bert = XLNetModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        elif PRE_TRAINED_MODEL_NAME[0:6] == 'openai':
            self.bert = OpenAIGPTModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 2)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        #print(output)
        hidden_states = output[0]
        # I learned the below code from this discussion: https://github.com/huggingface/transformers/issues/7540
        #pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        #print(pooled_output)
        pooled_output = hidden_states[:, 0, :]
        #pooled_output = output.pooler_output
        #output = self.drop(output.pooler_output)
        output = self.drop(pooled_output)
        output = self.relu(output)
        output = self.l1(output)
        output = self.l2(output)
        output = self.out(output)
        #print(output)
        #max = self.softmax(output)
        return output

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """
    This function trains the model for one epoch.
    :param model: Climate Identifier model
    :param data_loader: data to classify
    :param loss_fn: loss function specified in the global parameters
    :param optimizer: optimizer
    :param device: device
    :param scheduler: scheduler
    :param n_examples: number of examples
    :return: the accuracy of the predictions and the mean loss associated with the data loader classification
    """
    model = model.train()

    losses = []
    correct_predictions = 0

    # create a threshold of .65
    # if both values are below that, it will take a value of 0, since that is the first index, which is what we want
    #thresh = nn.Threshold(.95, 0)

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        #_, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)

        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def train_model(train_data_loader, length_x_train, test_data_loader, length_x_test, new_model_name, PRE_TRAINED_MODEL_NAME):
    """
    This function fine-tunes the entire model.
    :param train_data_loader: data loader of the training dataset
    :param length_x_train: length of the train dataset
    :param test_data_loader: data loader of the testing dataset
    :param length_x_test: length of the test dataset
    :param new_model_name: the name of the future fine-tuned model to load to make predictions
    :param PRE_TRAINED_MODEL_NAME: the name of the type of model used
    :return: nothing is returned but some graphs are saved in the Output folder and the new fine-tuned model is saved
    """
    model = ClimateIdentifier(PRE_TRAINED_MODEL_NAME)
    model = model.to(device)

    # Model Training Function
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    total_steps = len(train_data_loader) * N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print('Start Training...', end='')

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(N_EPOCHS):
            print(f'Epoch {epoch + 1}/{N_EPOCHS}')
            print('-' * 10)
            train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                loss,
                optimizer,
                device,
                scheduler,
                length_x_train)
            print(f'Train loss {train_loss} accuracy {train_acc}')
            test_acc, test_loss = eval_model(
                model,
                test_data_loader,
                loss,
                device,
                length_x_test)
            print(f'Test loss {test_loss} accuracy {test_acc}')
            print()
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
            history['test_acc'].append(test_acc)
            history['test_loss'].append(test_loss)
            if test_acc >= best_accuracy:
                torch.save(model.state_dict(), new_model_name)
                best_accuracy = test_acc

    print('done')

    # Plot the Results
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['test_acc'], label='test accuracy')
    plt.title('Training History for {}'.format(PRE_TRAINED_MODEL_NAME))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.show()

    # Save the results
    plt.savefig('output/accuracy_{}'.format(PRE_TRAINED_MODEL_NAME))

def model_validation(model, data_loader, n_examples):
    """
    This function performs model validation on the validation dataset
    :param model: Climate Identifier Model
    :param data_loader: validation data loader
    :param n_examples: number of sentences in the validation data loader
    :return: only prints the accuracy of the validation classifications
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)
    total_steps = len(data_loader) * N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    val_acc, _ = eval_model(model, data_loader, loss_fn, device, n_examples)
    print("Accuracy:", val_acc.item())

def eval_model(model, data_loader, loss_fn, device, n_examples):
    """
    This function evaluates the model.
    :param model: Climate Identifier model
    :param data_loader: data loader used to evaluate the model
    :param loss_fn: loss function, as specified in the global parameters
    :param device: device
    :param n_examples: number of sentences in the data loader
    :return: the accuracy of the predictions and the mean loss associated with the data loader classification
    """
    losses = []
    correct_predictions = 0

    # create a threshold of .65
    # if both values are below that, it will take a value of 0, since that is the first index, which is what we want
    #thresh = nn.Threshold(.95, 0)

    with torch.no_grad():
      for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        #_, preds = torch.max(thresh(torch.softmax(outputs, dim=1)), dim=1)
        _, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

    #print("Accuracy:", (correct_predictions.double() / n_examples).item())
    return correct_predictions.double() / n_examples, np.mean(losses)

def load_trained_model(model_path, PRE_TRAINED_MODEL_NAME):
    """
    This function loads the fine-tuned model to make predictions with.
    :param model_path: the file path where the fine-tuned model is stored
    :param PRE_TRAINED_MODEL_NAME: model type
    :return: the fine-tuned model as a Climate Identifier
    """
    print("Loading the fine-tuned model...", end='')
    model = ClimateIdentifier(PRE_TRAINED_MODEL_NAME)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    print("done")
    return model

def get_predictions(model, data_loader):
    """
    This function makes the predictions on the training dataset.
    :param model: Climate Identifier Model
    :param data_loader: data loader that contains all the training sentences to predict
    :return: this returns the sentences, the predictions, the predicted probabilities associated with the predictions, and the real value
    """
    print('Making Predictions on 10-K data...', end='')
    model = model.eval()
    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []

    # create a threshold of .65
    # if both values are below that, it will take a value of 0, since that is the first index, which is what we want
    #thresh = nn.Threshold(.95, 0)

    with torch.no_grad():
      for d in data_loader:
        texts = d["sentence"]
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        #print(thresh(torch.softmax(outputs, dim=1)))
        #print(texts)
        #print(torch.max(thresh(torch.softmax(outputs, dim=1)), dim=1))
        _, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        sentences.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(outputs)
        real_values.extend(targets)
    print('done')
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return sentences, predictions, prediction_probs, real_values

def model_metrics(pred, true_values, pred_probs, PRE_TRAINED_MODEL_NAME):
    """
    This function produces metrics on the success of the model
    :param pred: predictions made by the model
    :param true_values: true values from the model
    :param pred_probs: prediction probabilities
    :param PRE_TRAINED_MODEL_NAME: model type
    :return: the function returns nothing but a confusion matrix is saved in the Output folder
    """
    print(classification_report(true_values, pred, target_names=["Not Climate Related", "Climate Related"]))

    # Confusion Matrix
    cm = metrics.confusion_matrix(true_values, pred)
    print(cm)
    # Visualize
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', annot_kws={"size": 20});
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Confusion Matrix for {}'.format(PRE_TRAINED_MODEL_NAME)
    plt.title(all_sample_title, size = 25)
    plt.show()

    plt.savefig('output/confusion_matrix_{}'.format(PRE_TRAINED_MODEL_NAME))

def example_classifications(sentences, real_values, pred, index):
    """
    This function returns one example of a classification result from the model. This will be useful for visualizations.
    :param sentences: sentences
    :param real_values: real targets
    :param pred: prediction from the model
    :param index: the numerical index of the certain sentence in the list of sentences to visualize
    :return: nothing, but it prints the sentence
    """
    sentence = sentences[index]
    true_class = real_values[index]
    predicted_class = pred[index]

    print("\n".join(wrap(sentence)))
    print()
    print(f'True Climate Classification: {true_class}')
    print()
    print(f'Predicted Cliamte Classification: {predicted_class}')
    print()

def get_predictions_skinny(model, data_loader):
    """
    This function gets the predictions on the unlabeled 10-K sentences.
    I tried to implement this in parallel but did not get it to work.
    I have kept the parallel code commented out to refine further at a later date.
    :param model: Climate Identifier Model
    :param data_loader: data loader that contains the 10-K sentences
    :return: an array of the predictions
    """
    model = model.eval()
    #manager = multiprocessing.Manager()
    #predictions = manager.list()
    predictions = []
    #predictions = ['a'] * (len(data_loader)*BATCH_SIZE)
    # for i in range(len(data_loader)*BATCH_SIZE):
    #   predictions.append('a')

    # create a threshold of .65
    # if both values are below that, it will take a value of 0, since that is the first index, which is what we want
    thresh = nn.Threshold(.98, 0)

    t1 = time.perf_counter()
    count = 0
    with torch.no_grad():
      # COULD NOT GET PARALLISM TO WORK SADLY :(
      #with cf.ThreadPoolExecutor(max_workers=6) as executor:
        # executor.map(full_prediction_helper, (data_loader, model, predictions, thresh))
        #for i in executor.map(functools.partial(full_prediction_helper, model=model, predictions=predictions, thresh=thresh), data_loader):
         #   print(i)
         #   predictions.insert(i[0], i[1])
         #   del i
        #for future in cf.as_completed(some_preds):
        #    for i in range(0, len(future.result()[0])):
         #       predictions.insert(future.result()[0][i], future.result()[1][i])
      for d in data_loader:
        if count % (BATCH_SIZE*1000) == 0:
                t2 = time.perf_counter()
                print("{} sentences have been processed in {} minutes".format(count, round((t2 - t1) / 60, 2)))
        targets = d["targets"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(thresh(torch.softmax(outputs, dim=1)), dim=1)
        #_, preds = torch.max(torch.softmax(outputs, dim=1), dim=1)
        #print(torch.softmax(outputs, dim=1))
        #print(preds)
        #for i in range(0, len(preds)):
        #    predictions.insert(preds[i], targets[i])
        predictions.extend(preds)
        count += BATCH_SIZE
    #print(predictions)
    #for i in [1]*BATCH_SIZE:
        #if predictions[-i] == 'a':
           #predictions.pop()

    predictions = torch.stack(predictions).cpu()

    return predictions

def read_in_sentences():
    """
    This function reads in the text file of climate sentences extracted from 10-K files into an array.
    :return: an array of each sentence
    """
    climate_sentences = []
    with open("Domain-Agnostic-Sentence-Specificity-Prediction/dataset/data/ten_k_sentences.txt") as xh:
        # Read sentence file
        xlines = xh.readlines()
        number_of_sentences = len(xlines)
        print("Number of sentences to make predictions for:", number_of_sentences)
        for i in range(len(xlines)):
            climate_sentences.append(xlines[i].strip())

    return climate_sentences

def full_tokenize_and_create_loaders(x, PRE_TRAINED_MODEL_NAME):
    """
    This function creates the data loader for the full predictions on the 10-K sentences.
    :param x: an array of 10-K unlabeled sentences
    :param PRE_TRAINED_MODEL_NAME: model type
    :return: data loader with all 10-K sentences
    """
    print("Tokenize and create data loaders for the full prediction...", end='')
    if PRE_TRAINED_MODEL_NAME[0:4] == 'bert':
        tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    elif PRE_TRAINED_MODEL_NAME[0:7] == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    elif PRE_TRAINED_MODEL_NAME[0:5] == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    elif PRE_TRAINED_MODEL_NAME[0:6] == 'openai':
        tokenizer = OpenAIGPTTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    full_data_loader = create_data_loader(x, np.arange(0, len(x)).tolist(), tokenizer, BATCH_SIZE)
    print('done')

    return full_data_loader

def full_predictions_output(x, model):
    """
    This function calls the get_predictions_skinny function to make the predictions and then outputs the predictions of the 10-K sentences to a text file.
    The code was taken from here https://stackoverflow.com/questions/39717828/python-combine-two-text-files-into-one-line-by-line
    :param x: data loader of 10-K sentences
    :param model: Climate Identifier Model
    :return: nothing, but outputs a text file of the predictions
    """
    # Make predictions
    print('Making Predictions on 10-K data...', end='')
    preds = get_predictions_skinny(x, model)
    print("done")
    with open("Domain-Agnostic-Sentence-Specificity-Prediction/climate_predictions.txt", "w") as wh:

        # Write predictions
        for i in range(len(preds)):
            wh.write(str(preds[i].numpy()) + '\n')

#==============================================================================#
# Code to attempt a parallel implementation of the 10-K predictions... not yet working
#==============================================================================#
#torch.multiprocessing.set_sharing_strategy('file_system')
def full_prediction_helper(d, model, predictions, thresh):
    targets = d["targets"].to(device)
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    _, preds = torch.max(thresh(torch.softmax(outputs, dim=1)), dim=1)
    #for i in range(0, len(preds)):
    #    predictions.insert(preds[i], targets[i])
    return (preds, targets)

# print
def printing(d, model):
    print(d)
    print(model)

def wrapper(tup):
    return full_prediction_helper(*tup)


