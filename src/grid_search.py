import torch
import torch.nn as nn
from torch.utils.data import random_split

from skorch import NeuralNetClassifier
from skorch.dataset import ValidSplit
from sklearn.model_selection import GridSearchCV

from classifier import Classifier
from dataset import Pets
import utils

CLASSIFICATION_MODE = "multi_class"
BATCH_SIZE = 70
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

if __name__ == '__main__':
    # Create hyperparam search folder.
    search_folder = utils.create_search_run()

    classifier = Classifier(CLASSIFICATION_MODE)
    # Learning parameters. 
    lr = 0.001
    epochs = 20
    device = 'cpu'
    print(f"Computation device: {device}\n")

    # Loss function. Required for defining `NeuralNetClassifier`
    criterion = nn.CrossEntropyLoss()
    
    # Instance of `NeuralNetClassifier` to be passed to `GridSearchCV` 
    net = NeuralNetClassifier(
        module=classifier, max_epochs=epochs,
        optimizer=torch.optim.Adam,
        criterion=criterion,
        lr=lr, verbose=1, train_split=ValidSplit(0.1)
    )

    root_dir = "./data/images/"
    dataset = Pets(
        root_dir=root_dir,
        transform=classifier.input_transform,
        classification_mode=classifier.classification_mode
    )

    val_dataset = Pets(
        root_dir=root_dir,
        transform=classifier.test_transform,
        classification_mode=classifier.classification_mode
    )


    try:
        train_data, _, _ = random_split(dataset, [TRAIN_SPLIT, VAL_SPLIT,TEST_SPLIT],torch.Generator().manual_seed(69))
        _, val_data, test_data = random_split(val_dataset, [TRAIN_SPLIT, VAL_SPLIT,TEST_SPLIT],torch.Generator().manual_seed(69))
    except:
        train_split = int(TRAIN_SPLIT * len(dataset))
        val_split = int(VAL_SPLIT * len(dataset))
        test_split = int(len(dataset)- train_split-val_split)

        train_data, _, _ = random_split(dataset, [train_split, val_split, test_split],torch.Generator().manual_seed(69))
        _, val_data, test_data = random_split(val_dataset, [train_split, val_split, test_split],torch.Generator().manual_seed(69))

    # Get the training and validation data loaders.
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )


    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE , shuffle=False
    )
    
    params = {
    'lr': [0.001, 0.01, 0.005, 0.0005, 0.0001, 1e-5],
    'max_epochs': list(range(20, 55, 5)),
    }
    """
    Define `GridSearchCV`.
    4 lrs * 7 max_epochs * 5 lambda * 2 CVs = 280 fits.
    """
    gs = GridSearchCV(
        net, params, refit=False, scoring='accuracy', verbose=1, cv=2
    )
    counter = 0
    # Run each fit for 2 batches. So, if we have `n` fits, then it will
    # actually for `n*2` times. We have 672 fits, so total, 
    # 672 * 2 = 1344 runs.
    search_batches = 2
    """
    This will run `n` (`n` is calculated from `params`) number of fits 
    on each batch of data, so be careful.
    If you want to run the `n` number of fits just once, 
    that is, on one batch of data,
    add `break` after this line:
        `outputs = gs.fit(image, labels)`
    Note: This will take a lot of time to run
    """
    for i, (img_batch, target_batch) in enumerate(train_dataloader):
        counter += 1
        img_batch = img_batch.to(device)
        target_batch = target_batch.to(device)
        outputs = gs.fit(img_batch, target_batch)
        # GridSearch for `search_batches` number of times.
        if counter == search_batches:
            break
    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
    # save_best_hyperparam(gs.best_score_, f"../outputs/{search_folder}/best_param.yml")
    # save_best_hyperparam(gs.best_params_, f"../outputs/{search_folder}/best_param.yml")