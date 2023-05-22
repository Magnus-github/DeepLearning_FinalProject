import itertools
import os
from datetime import datetime
from typing import Tuple
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
import torch
#import wandb
from PIL import Image
from torch import nn
from torch.utils.data import random_split

from classifier import Classifier
from dataset import Pets
import utils

import sounddevice as sd
import soundfile as sf

CLASSIFICATION_MODE = "multi_class"
VALIDATION_ITERATION = 20
VALIDATE = True
NUM_ITERATIONS = 1000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

learning_rates = [1e-3, 5e-4, 1e-4]
weight_decays = [0.0005, 0.0001, 0.00005]
batch_sizes = [128, 256, 512]


best_accuracy = 0.0
best_parameters = {}

results_dir = "./outputs/grid_search_results"
os.makedirs(results_dir, exist_ok=True)


def compute_loss(
    prediction_batch: torch.Tensor, target_batch: torch.Tensor
) -> Tuple[torch.Tensor]:
    """Compute loss between predicted tensor and target tensor.

    Args:
        prediction_batch: Batched predictions. Shape (N,k).
        target_batch: Batched targets. shape (N,k).

    Returns:
        cross-entropy loss
    """
    if CLASSIFICATION_MODE == "multi_class":
        loss = torch.nn.functional.cross_entropy(prediction_batch,target_batch.type(torch.float))
    elif CLASSIFICATION_MODE == "binary":
        loss = torch.nn.functional.binary_cross_entropy(prediction_batch,target_batch.type(torch.float))
    return loss

def compute_accuracy(prediction: torch.Tensor, ground_truth: torch.Tensor):
    correct = prediction == ground_truth
    num_correct = torch.sum(correct).item()
    acc = num_correct/len(ground_truth)
    return acc



def grid_search(device: str = "cpu") -> None:
    """Train the network.

    Args:
        device: The device to train on ('cpu' or 'cuda').
    """
    global best_accuracy, best_parameters
    # wandb.init(project="Object_detection_wAugmentation-1")

    # Init model
    classifier = Classifier(classification_mode=CLASSIFICATION_MODE).to(device)

    # wandb.watch(classifier)
    root_dir = "."
    if "data" in os.listdir(os.curdir):
        root_dir = "./data/images/"
    else:
        root_dir = "../data/images/"

    dataset = Pets(
        root_dir=root_dir,
        transform=classifier.input_transform,
        classification_mode=CLASSIFICATION_MODE
    )

    val_dataset = Pets(
        root_dir=root_dir,
        transform=classifier.test_transform,
        classification_mode=CLASSIFICATION_MODE
    )


    train_data, _ = random_split(dataset, [TRAIN_SPLIT, VAL_SPLIT],torch.Generator().manual_seed(69))
    _, val_data = random_split(val_dataset, [TRAIN_SPLIT, VAL_SPLIT],torch.Generator().manual_seed(69))

    num_search = 1
    # ITERATING OVER BATCH SIZES
    for bs in batch_sizes:

        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=bs, shuffle=True
        )


        val_dataloader = torch.utils.data.DataLoader(
            val_data, batch_size=bs , shuffle=False
        )

        # ITERATING OVER WEIGHT DECAYS
        for wd in weight_decays:
            # ITERATING OVER LEARNING RATES
            for lr in learning_rates:

                # Create a directory for the current parameter setting
                param_dir = os.path.join(results_dir, f"param_{num_search}")
                os.makedirs(param_dir, exist_ok=True)
                
                # Save the parameter settings in a text file
                settings_file = os.path.join(param_dir, "settings.txt")
                with open(settings_file, "w") as f:
                    f.write(f"Learning Rate: {lr}\n")
                    f.write(f"Weight Decay: {wd}\n")
                    f.write(f"Batch Size: {bs}\n")

                # init optimizer
                optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2, eta_min=0.8*lr, verbose=False)

                classifier.eval()

                t = []
                train_losses = []
                val_losses = []
                train_accs = []
                val_accs = []
                unfreeze = 1
                running = True

                iter_p_epoch = len(train_data)/bs

                print("Running {} epochs...".format(int(NUM_ITERATIONS/int(iter_p_epoch))))
                print("Training started...")

                # waveform, sample_rate = sf.read("./data/Startingtraining.wav", dtype='float32')
                # sd.play(waveform, sample_rate)
                # sd.wait()

                current_iteration = 1
                val_iteration = 1
                while (current_iteration <= NUM_ITERATIONS) and running:
                    for img_batch, target_batch in train_dataloader:

                        img_batch = img_batch.to(device)
                        target_batch = target_batch.to(device)
                        if CLASSIFICATION_MODE == "binary":
                            target_onehot = nn.functional.one_hot(target_batch, 2)
                        elif CLASSIFICATION_MODE == "multi_class":    
                            target_onehot = nn.functional.one_hot(target_batch, 37)
                        

                        # run network (forward pass)
                        out = classifier(img_batch)
                        if CLASSIFICATION_MODE == "binary":
                            out = torch.nn.functional.softmax(out, 1)

                        loss = compute_loss(out, target_onehot)

                        # optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        print(
                            "Iteration: {}, loss: {}".format(
                                current_iteration, loss.item()),
                        )

                        # Validate and plot every N iterations
                        if VALIDATE:
                            if current_iteration % VALIDATION_ITERATION == 0:
                                val_loss, val_acc = validate(classifier, bs, val_dataloader, device)
                                loss = loss.to("cpu").detach().numpy()
                                val_loss = val_loss.to("cpu").detach().numpy()
                                # val_acc = val_acc.to("cpu").detach().numpy()
                                    
                                # train_accs.append(train_acc)
                                val_accs.append(val_acc)
                                train_losses.append(loss)
                                val_losses.append(val_loss)

                                if len(val_accs) > 1:
                                    print("---------------------------------------------------------------------------")
                                    print("Validation loss: {}; Diff: {}".format(val_loss, val_loss-val_losses[-2]))
                                    print("Validation acc: {}; Diff: {}".format(val_acc, val_acc-val_accs[-2]))
                                    print("Training loss diff to last val step: {}".format(loss-train_losses[-2]))

                                    # if abs(val_acc-val_accs[-2]) < 0.005:
                                        # if unfreeze < LAYERS_TO_UNFREEZE+1:
                                        #     print("Unfreezing last {} layers of the pretrained model.".format(unfreeze))
                                        #     for param in list(classifier.features.parameters())[:-unfreeze]:
                                        #         param.requires_grad = True
                                        #     # for child in classifier.features.children():                                
                                        #     #     for param in list(child.parameters())[:-unfreeze]:
                                        #     #         param.requires_grad = True
                                        #     unfreeze+=1
                                    if (val_loss-val_losses[-2]) > 0.0001:
                                        print("Stopping training as validation loss is increasing.")
                                        running = False
                                    print("---------------------------------------------------------------------------")

                                val_iteration += 1

                        current_iteration += 1
                        if (current_iteration > NUM_ITERATIONS) or not running:
                            running = False
                            break
            
                    # waveform, sample_rate = sf.read("./data/Ichhabefertig.wav", dtype='float32')
                    # sd.play(waveform, sample_rate)
                    # sd.wait()
                    print("\nTraining completed")


                    t = np.linspace(0, current_iteration, num=len(train_losses))
                    plt.figure()
                    plt.plot(t,train_losses,label="Training loss")
                    plt.plot(t,val_losses, label="Validation loss")
                    plt.ylabel("Loss")
                    plt.xlabel("t")
                    plt.title("Loss function")
                    plt.legend()
                    plt.savefig(param_dir+"/Losses.pdf", format="pdf", bbox_inches="tight")

                    plt.figure()
                    # plt.plot(t,train_accs,label="Training accuracy")
                    plt.plot(t,val_accs, label="Validation accuracy")
                    plt.ylabel("Accuracy")
                    plt.xlabel("t")
                    plt.title("Validation accuracy")
                    plt.legend()
                    plt.savefig(param_dir+"/Accuracies.pdf", format="pdf", bbox_inches="tight")

                    if val_accs[-1] > best_accuracy:
                        best_accuracy = val_accs[-1]
                        best_parameters["batch size"] = bs
                        best_parameters["Learning Rate"] = lr
                        best_parameters["Weight decay"] = wd
                        print("Current best accuracy: ",best_accuracy)
                        print("Parameters: ", best_parameters)

                    num_search += 1


def validate(
    classifier: Classifier,
    bs: int,
    val_dataloader: torch.utils.data.DataLoader,
    device: str,
) -> None:
#     """Compute validation metrics and log to wandb.

#     Args:
#         detector: The detector module to validate.
#         val_dataloader: The dataloader for the validation dataset.
#         device: The device to run validation on.
#     """
    classifier.eval()
    
    with torch.no_grad():
        count = total_loss = acc = 0
        for val_img_batch, val_target_batch in val_dataloader:
            val_img_batch = val_img_batch.to(device)
            val_target_batch = val_target_batch.to(device)
            if CLASSIFICATION_MODE == "binary":
                val_target_onehot = nn.functional.one_hot(val_target_batch, 2)
            elif CLASSIFICATION_MODE == "multi_class":    
                val_target_onehot = nn.functional.one_hot(val_target_batch, 37)
            val_out = classifier(val_img_batch)
            loss = compute_loss(val_out, val_target_onehot)
            val_out = torch.argmax(val_out, 1)
            acc += compute_accuracy(val_out, val_target_batch)
            total_loss +=loss
            count += len(val_img_batch) / bs

    classifier.train()

    return total_loss/count, acc/count


if __name__ == "__main__":
    grid_search("cuda")
    print("DONE WITH SEARCH!!!")
    print("Best val accuracy: ", best_accuracy)
    print("Parmaters: ", best_parameters)