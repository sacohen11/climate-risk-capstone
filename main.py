#################################################################
# Title: Main Research Pipeline
# Author: Sam Cohen

# Notes:
# This script runs the main research pipeline to obtain the results as
# highlighted in the paper.
##################################################################

#Extract sentences in section 1A. of 10-K documents of companies in the S&P 500
import extract_10_k
sp500 = extract_10_k.companies_sp_500()
ciks = extract_10_k.get_cik_translator()
#ciks = extract_10_k.getCIKs(sp500)

dict = extract_10_k.ten_k_dictionary(sp500, ciks)
extract_10_k.write_to_txt(dict, cik_crosswalk=ciks, sp500=sp500)

extract_10_k.count_number_of_companies()

# create random sentences to use in training the model
import random_sentences
random_sentences.random_sentences()

# run climate model
import climate_identification_models
x_train, y_train, x_test, y_test, x_val, y_val = climate_identification_models.load_training_data()
train_data_loader, test_data_loader, val_data_loader = climate_identification_models.tokenize_and_create_loaders(x_train, y_train, x_test, y_test, x_val, y_val, 'bert-base-cased')

for model_type in ['bert-base-cased','bert-base-uncased', 'xlnet-base-cased', 'roberta-base', 'openai-gpt']:

    # Train model... uncomment if you'd like to train
    climate_identification_models.train_model(train_data_loader, len(x_train), test_data_loader, len(x_test), 'best_model_t{}.bin'.format(model_type), model_type)

    model = climate_identification_models.load_trained_model('best_model_t{}.bin'.format(model_type), model_type)
    climate_identification_models.model_validation(model, val_data_loader, len(x_val))
    sentences, predictions, prediction_probs, real_values = climate_identification_models.get_predictions(model, val_data_loader)
    climate_identification_models.model_metrics(predictions, real_values, prediction_probs, model_type)
    climate_identification_models.example_classifications(sentences, real_values, predictions, 11)

# Bert Cased is the best model... I will use that to make the predictions
model_type = 'bert-base-cased'
x = climate_identification_models.read_in_sentences()
full_data_loader = climate_identification_models.full_tokenize_and_create_loaders(x, 'bert-base-cased')
model = climate_identification_models.load_trained_model('best_model_t{}.bin'.format(model_type), model_type)
climate_identification_models.full_predictions_output(model, full_data_loader)

#run specificity model
import os

# chdir
os.chdir("./Domain-Agnostic-Sentence-Specificity-Prediction/")
# train
os.system("python3 train.py --gpu_id 0 --test_data ten_k")
# test
os.system("python3 test.py --gpu_id 0 --test_data ten_k")

# chdir back
os.chdir("..")

# combine the predictions of both models
import combine_preds_and_analyze
combine_preds_and_analyze.combine_predictions()
combine_preds_and_analyze.eda_plots()