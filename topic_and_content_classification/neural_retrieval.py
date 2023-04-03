"""
neural_retrieval.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for neural text retrieval.
"""

# import packages

import numpy as np
import torch, torchvision

# import functions form other files in this project


# functions for creating and training a neural retrieval model

def create_model(model_type: str) -> torch.nn.Module:
    """
    Create a neural retrieval model.

    Parameters
    ----------
    model_type : The type of model to create.

    Returns
    -------
    model : The created model.
    """
    assert model_type in ["bert", "MLP"], "The model type must be either 'bert' or 'MLP'."

    if model_type == "bert":
        model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    elif model_type == "MLP":
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    return model


def load_data(corpus_folder):
    """
    To be implemented. 
    """
    pass


def train_model(model: torch.nn.Module, X_train: np.ndarray, Y_train: np.ndarray, 
                X_val: np.ndarray, Y_val: np.ndarray, batch_size: int, 
                n_epochs: int, learning_rate: float) -> torch.nn.Module:
    """
    Train a neural retrieval model.

    Parameters
    ----------
    model : The model to train.

    X_train : The training data.

    Y_train : The training labels.

    X_val : The validation data.

    Y_val : The validation labels.

    batch_size : The batch size to use.

    n_epochs : The number of epochs to train for.

    learning_rate : The learning rate to use.

    Returns
    -------
    model : The trained model.
    """
    # create a training dataset
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))

    # create a validation dataset
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))

    # create a training data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # create a validation data loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # create a loss function
    loss_function = torch.nn.BCEWithLogitsLoss()

    # train the model
    for epoch in range(n_epochs):
        for X_batch, Y_batch in train_loader:
            # compute the model output
            Y_hat = model(X_batch)

            # compute the loss
            loss = loss_function(Y_hat, Y_batch)

            # compute the gradients
            loss.backward()

            # update the model parameters
            optimizer.step()

            # reset the gradients
            optimizer.zero_grad()

        # evaluate the model on the validation set
        Y_hat = model(X_val)
        loss = loss_function(Y_hat, Y_val)
        print("Epoch: {}, Validation loss: {}".format(epoch, loss))

    return model


def neural_retrieval(model: torch.nn.Module, corpus_folder: str) -> list:
    """
    Perform neural retrieval using a pretrained model.

    Parameters
    ----------
    model : The pretrained model to use.

    corpus_folder : The folder containing the corpus.

    Returns
    -------
    documents : The retrieved documents.
    """
    # load the data
    X, Y = load_data(corpus_folder)

    # create a dataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(X))

    # create a data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # perform retrieval by computing the model output for each document
    documents = []
    for X_batch in data_loader:
        Y_hat = model(X_batch)
        if Y_hat > 0.5:
            documents.append(X_batch)

    return documents