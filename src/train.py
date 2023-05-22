"""Training script for classifier."""
import argparse
import copy
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

from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

from classifier import Classifier
from dataset import Pets
import utils

CLASSIFICATION_MODE = "multi_class"
VALIDATION_ITERATION = 20
VALIDATE = True
NUM_ITERATIONS = 2000
LEARNING_RATE = 1e-5
LEARNING_RATE_MAX = 1e-2
BATCH_SIZE = 10
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
LAMBDA = 0.05
LAYERS_TO_UNFREEZE = 5

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
    acc = 1 - (prediction-ground_truth).count_nonzero()/len(ground_truth)
    return acc


def train(device: str = "cpu") -> None:
    """Train the network.

    Args:
        device: The device to train on ('cpu' or 'cuda').
    """

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


    try:
        train_data, _, _ = random_split(dataset, [TRAIN_SPLIT, VAL_SPLIT,TEST_SPLIT],torch.Generator().manual_seed(69))
        _, val_data, test_data = random_split(val_dataset, [TRAIN_SPLIT, VAL_SPLIT,TEST_SPLIT],torch.Generator().manual_seed(69))
    except:
        train_split = int(TRAIN_SPLIT * len(dataset))
        val_split = int(VAL_SPLIT * len(dataset))
        test_split = int(len(dataset)- train_split-val_split)

        train_data, _, _ = random_split(dataset, [train_split, val_split, test_split],torch.Generator().manual_seed(69))
        _, val_data, test_data = random_split(val_dataset, [train_split, val_split, test_split],torch.Generator().manual_seed(69))

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )


    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE , shuffle=False
    )

    test_dataloader = torch.utils.data.DataLoader(
         test_data, batch_size=100, shuffle=False
    )

    test = Pets(
        root_dir=root_dir,
        classification_mode=classifier.classification_mode
    )

    test_lb, test_unlb = random_split(test, [0.2,0.8])
    print(test_lb)
    exit()

    # training params
    # wandb.config.max_iterations = NUM_ITERATIONS
    # wandb.config.learning_rate = LEARNING_RATE
    # wandb.config.weight_pos = WEIGHT_POS
    # wandb.config.weight_neg = WEIGHT_NEG
    # wandb.config.weight_reg = WEIGHT_REG

    # run name (to easily identify model later)
    # time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    # run_name = wandb.config.run_name = "det_{}".format(time_string)

    # init optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=LAMBDA)
    # optimizer = torch.optim.SGD(classifier.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=LAMBDA)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LEARNING_RATE, max_lr=LEARNING_RATE_MAX, mode='triangular2')
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=LEARNING_RATE)

    classifier.eval()

    t = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    unfreeze = 1
    running = True


    print("Training started...")

    current_iteration = 1
    val_iteration = 1
    while (current_iteration <= NUM_ITERATIONS) and running:
        for img_batch, target_batch in train_dataloader:
            
            idxs = torch.where(target_batch==4)[0].tolist()
            op_idxs = torch.where(target_batch!=4)[0].tolist()

            op_imgs = img_batch[op_idxs,:,:,:]
            op_target = target_batch[op_idxs]

            imgs = img_batch[idxs,:,:,:]
            print(imgs.size())
            print(op_imgs.size())
            print(idxs)
            print(op_target)
            exit()
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
            # scheduler.step()
            
            # wandb.log(
            #     {
            #         "total loss": loss.item(),
            #         "loss pos": pos_mse.item(),
            #         "loss neg": neg_mse.item(),
            #         "loss reg": reg_mse.item(),
            #     },
            #     step=current_iteration,
            # )

            print(
                "Iteration: {}, loss: {}".format(
                    current_iteration, loss.item()),
            )

            # Validate and plot every N iterations
            if VALIDATE:
                if current_iteration % VALIDATION_ITERATION == 0:
                    val_loss, val_acc = validate(classifier, val_dataloader, device)
                    loss = loss.to("cpu").detach().numpy()
                    val_loss = val_loss.to("cpu").detach().numpy()
                    val_acc = val_acc.to("cpu").detach().numpy()
                    # with torch.no_grad():
                    #     count = train_acc = 0
                    #     for i, (train_imgs, train_target) in enumerate(train_dataloader):
                    #         train_imgs = train_imgs.to(device)
                    #         train_target = train_target.to(device)
                    #         train_out = classifier(train_imgs)
                    #         train_out = torch.argmax(train_out, 1)
                    #         train_acc += compute_accuracy(train_out, train_target)
                    #         count += len(train_imgs) / BATCH_SIZE
                    # train_acc = train_acc/count
                    # train_acc = train_acc.to("cpu").detach().numpy()
                        
                    # train_accs.append(train_acc)
                    val_accs.append(val_acc)
                    train_losses.append(loss)
                    val_losses.append(val_loss)

                    if len(val_accs) > 1:
                        if abs(val_acc-val_accs[-2]) < 0.005:
                            if unfreeze < LAYERS_TO_UNFREEZE+1: 
                                print("Unfreezing last {} layers of the pretrained model.".format(unfreeze))
                                for child in classifier.features.children():                                
                                    for param in list(child.parameters())[:-unfreeze]:
                                        param.requires_grad = True
                                unfreeze+=1
                        if abs(val_acc-val_accs[-2]) < 0.0005:
                            print("Stopping training as validation accuracy doesn't increase anymore.")
                            running = False

                    # update_plot(trainloss_plot, valloss_plot, loss, val_loss, val_iteration)
                    val_iteration += 1

            current_iteration += 1
            if (current_iteration > NUM_ITERATIONS) or not running:
                running = False
                break

    print("\nTraining completed (max iterations reached)")

    classifier.eval()
    acc = 0
    all = 0
    with torch.no_grad():
        for i, (test_imgs, test_target) in enumerate(test_dataloader):
            test_imgs = test_imgs.to(device)
            test_target = test_target.to(device)
            test_out = classifier(test_imgs)
            test_out = torch.argmax(test_out, 1)
            acc += compute_accuracy(test_out, test_target)
            print(acc/(i+1))
            all = i+1
    acc = acc/all
    print("FINAL: ",acc)


    # model_path = "{}.pt".format(run_name)
    # utils.save_model(classifier, model_path)
    # wandb.save(model_path)

    # print("Model weights saved at {}".format(model_path))
    # t = range(0,current_iteration+1,int((current_iteration+1)/(len(train_losses)-1)))
    t = np.linspace(0, current_iteration, num=len(train_losses))
    plt.figure()
    plt.plot(t,train_losses,label="Training loss")
    plt.plot(t,val_losses, label="Validation loss")
    plt.ylabel("Loss")
    plt.xlabel("t")
    plt.title("Loss function")
    plt.legend()
    # plt.show()
    plt.savefig("./data/plots/Losses.pdf", format="pdf", bbox_inches="tight")

    plt.figure()
    # plt.plot(t,train_accs,label="Training accuracy")
    plt.plot(t,val_accs, label="Validation accuracy")
    plt.ylabel("Loss")
    plt.xlabel("t")
    plt.title("Accuracy")
    plt.legend()
    # plt.show()
    plt.savefig("./data/plots/Accuracies.pdf", format="pdf", bbox_inches="tight")


def validate(
    classifier: Classifier,
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
            count += len(val_img_batch) / BATCH_SIZE
    #         wandb.log(
    #             {
    #                 "total val loss": (loss / count)
    #             },
    #             step=current_iteration,
    #         )
    # print(
    #     "Validation: {}, validation loss: {}".format(
    #         current_iteration, loss / count
    #     ),
    # )

    classifier.train()

    return total_loss/count, acc/count


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # device = parser.add_mutually_exclusive_group(required=True)
    # device.add_argument("--cpu", dest="device",
    #                     action="store_const", const="cpu")
    # device.add_argument("--gpu", dest="device",
    #                     action="store_const", const="cuda")
    # args = parser.parse_args()
    # train(args.device)
    train("cuda")
    
