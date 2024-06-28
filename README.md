# Assignment 4: ELMO

## Name: Ishan Kavathekar
## Roll no: 2022121003


This repository contains code and models for training a ELMo model for language modeling task, as well as training classification models using these word embeddings for the given downstream task.

## Assumptions:
- Split the concatenated words ( \\\ ).
- Remove punctuation and special characters.
- If the frequency of a word is less than 5, replace it with the `<UNK>` tag.

## Files

### Source Code
 - <code>ELMO.py</code>: Train the biLSTM on the language modelling tas
 - <code>classification.py</code>:  Train the classifier for the downstream task using the word representations from the ELMo model

### Pretrained Models
- <code>bilstm.pt</code>: Saved biLSTM model.
-  <code>classifier.pt</code>: Saved classifier model used for the downstream task.


### Report
- `report.pdf`: PDF document containing:
  - Hyperparameters used to train the models.
  - Corresponding graphs and evaluation metrics.
  - Analysis of the results.

## Usage
- Use <code>ELMO.py</code> to train the ELMO model on the language modeling task. User can specify the model name in the script and save the model.
- Use <code>classifier.py</code> to train the model on the downstream task, plot the plots and get the classification report.



All the experimented models and embeddings can be found [here](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/ishan_kavathekar_research_iiit_ac_in/Eqa5GTOeDEJOpwFyWVALbl8BfGcgHxgO2fbX9swRPHHHzw?e=tagQJ6)
