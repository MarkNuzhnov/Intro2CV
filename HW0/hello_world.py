import numpy as np


def predict(img, model_path):
    with open(model_path, 'rb') as f:
        model = np.load(f)

    mean, std = model[0], model[1]

    return (img - model[0]) / model[1]


def train(img, save_model_path):
    with open(save_model_path, 'wb') as f:
        np.save(f, np.array([img.mean(), img.std()]))