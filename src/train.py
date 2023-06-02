"""Training script for classifier."""
import argparse
import copy
import os
from datetime import datetime
from typing import Tuple
from torchvision import transforms


import matplotlib.pyplot as plt
import torch
#import wandb
from PIL import Image
from torch import nn
from torch.utils.data import random_split

from classifier import Classifier
from dataset import Pets
import utils


VALIDATION_ITERATION = 20
NUM_ITERATIONS = 100
LEARNING_RATE = 1e-5
BATCH_SIZE = 95
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1


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
    
    loss = torch.nn.functional.cross_entropy(prediction_batch,target_batch.type(torch.float))
    return loss


def train(device: str = "cpu") -> None:
    """Train the network.

    Args:
        device: The device to train on ('cpu' or 'cuda').
    """

    # wandb.init(project="Object_detection_wAugmentation-1")

    # Init model
    classifier = Classifier(classification_mode="multi_class").to(device)

    # wandb.watch(classifier)
    #print(os.listdir(os.curdir))
    root_dir = "."
    if "data" in os.listdir(os.curdir):
        root_dir = "./data/images/"
    else:
        root_dir = "../data/images/"
<<<<<<< HEAD
    datasettrain = Pets(
=======
    dataset_train = Pets(
>>>>>>> 31d700cc7729bdf5ffd01f3ad642ed981cb0cca1
        root_dir=root_dir,
        transform=classifier.input_transform_training,
        classification_mode="binary"
    )
    dataset_test = Pets(
	    root_dir=root_dir,
	    transform=classifier.input_transform_testing,
	    classification_mode="binary"
    )
    datasettest = Pets(
        root_dir=root_dir,
        transform=classifier.test_transform,
        classification_mode="multi_class"
    )

    try:
<<<<<<< HEAD
        train_data, val_data, test_data = random_split(datasettrain, [TRAIN_SPLIT, VAL_SPLIT,TEST_SPLIT])
    except:
        train_split = int(TRAIN_SPLIT * len(datasettrain))
        val_split = int(VAL_SPLIT * len(datasettrain))
        test_split = int(len(datasettrain)- train_split-val_split)

        train_data, _, _ = random_split(datasettrain, [train_split, val_split, test_split])
        _,val_data, test_data= random_split(datasettest, [train_split, val_split, test_split])
=======
        train_data, _, _ = random_split(dataset_train, [TRAIN_SPLIT, VAL_SPLIT,TEST_SPLIT],
                                                       generator=torch.Generator().manual_seed(42))
        _, val_data, test_data = random_split(dataset_test, [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
                                                       generator=torch.Generator().manual_seed(42))
    except:
        TRAIN_SPLIT = int(TRAIN_SPLIT * len(dataset_train))
        VAL_SPLIT = int(VAL_SPLIT * len(dataset_test))
        TEST_SPLIT = int(len(dataset_test)- TRAIN_SPLIT-VAL_SPLIT)

        train_data, _, _ = random_split(dataset_train, [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
                                        generator=torch.Generator().manual_seed(42))
        _, val_data, test_data = random_split(dataset_test, [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT],
                                              generator=torch.Generator().manual_seed(42))



>>>>>>> 31d700cc7729bdf5ffd01f3ad642ed981cb0cca1

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )


    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE , shuffle=False
    )

    test_dataloader = torch.utils.data.DataLoader(
         test_data, batch_size=100, shuffle=False
    )

    # training params
    # wandb.config.max_iterations = NUM_ITERATIONS
    # wandb.config.learning_rate = LEARNING_RATE
    # wandb.config.weight_pos = WEIGHT_POS
    # wandb.config.weight_neg = WEIGHT_NEG
    # wandb.config.weight_reg = WEIGHT_REG

    # run name (to easily identify model later)
    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    # run_name = wandb.config.run_name = "det_{}".format(time_string)

    # init optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    # load test images
    # these will be evaluated in regular intervals
    classifier.eval()

    # image, target = dataset.__getitem__(0)
    # images = torch.zeros((1,image.size[0], image.size[1], image.size[2]))
    # out, features = detector(images)
    # print(features.size)
    # print(out.size)
    #exit()
    test_images = []
    show_test_images = False
    directory = "./data/test_images"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    # for file_name in sorted(os.listdir(directory)):
    #     if file_name.endswith(".jpeg"):
    #         file_path = os.path.join(directory, file_name)
    #         test_image = Image.open(file_path)
    #         torch_image, _ = detector.input_transform(test_image, [])
    #         test_images.append(torch_image)

    if test_images:
        test_images = torch.stack(test_images)
        test_images = test_images.to(device)
        show_test_images = True

    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # dataset.transforms = None
    # for i in range(0, 700,1):
    #     test = dataset
    #     image, target = dataset.__getitem__(i)
    #     fig, ax = plt.subplots()
    #     print(target)
    #     x,y,w,h = target[0]["bbox"]
    #     rect = patches.Rectangle((x,y), w,h, linewidth=1, edgecolor="r", facecolor="none")
    #     ax.imshow(image)
    #     ax.add_patch(rect)
    #     plt.show()
    # exit("forced exit")


    print("Training started...")

    current_iteration = 1
    while current_iteration <= NUM_ITERATIONS:
        for img_batch, target_batch in train_dataloader:

            
            img_batch = img_batch.to(device)
            target_batch = target_batch.to(device)
            target_onehot = nn.functional.one_hot(target_batch, 37)
            

            # run network (forward pass)
            out = classifier(img_batch)

            loss = compute_loss(out, target_onehot)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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

            # Validate every N iterations
            # if current_iteration % VALIDATION_ITERATION == 0:
            #     validate(classifier, val_dataloader, current_iteration, device)

            # # generate visualization every N iterations
            # if current_iteration % 250 == 0 and show_test_images:
            #     classifier.eval()
            #     with torch.no_grad():
            #         out = classifier(test_images).cpu()

            #         for i, test_image in enumerate(test_images):
            #             figure, ax = plt.subplots(1)
            #             plt.imshow(test_image.cpu().permute(1, 2, 0))
            #             plt.imshow(
            #                 out[i, 4, :, :],
            #                 interpolation="nearest",
            #                 extent=(0, 640, 480, 0),
            #                 alpha=0.7,
            #             )

            #             wandb.log(
            #                 {"test_img_{i}".format(i=i): figure}, step=current_iteration
            #             )
            #             plt.close()
            #     classifier.train()

            current_iteration += 1
            if current_iteration > NUM_ITERATIONS:
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
            acc += 1 - (test_out-test_target).count_nonzero()/len(test_target)
            print(acc/(i+1))
            all = i+1
    acc = acc/all
    print("FINAL: ",acc)


    # model_path = "{}.pt".format(run_name)
    # utils.save_model(classifier, model_path)
    # wandb.save(model_path)

    # print("Model weights saved at {}".format(model_path))


def validate(
    classifier: Classifier,
    val_dataloader: torch.utils.data.DataLoader,
    current_iteration: int,
    device: str,
) -> None:
#     """Compute validation metrics and log to wandb.

#     Args:
#         detector: The detector module to validate.
#         val_dataloader: The dataloader for the validation dataset.
#         current_iteration: The current training iteration. Used for logging.
#         device: The device to run validation on.
#     """
    classifier.eval()
    
    with torch.no_grad():
        count = total_loss = 0
        
        for val_img_batch, val_target_batch in val_dataloader:
            val_img_batch = val_img_batch.to(device)
            val_target_batch = val_target_batch.to(device)
            val_target_onehot = nn.functional.one_hot(val_target_batch)
            val_out = classifier(val_img_batch)
            loss = compute_loss(val_out, val_target_onehot)
            total_loss +=loss

            count += len(val_img_batch) / BATCH_SIZE
    #         wandb.log(
    #             {
    #                 "total val loss": (loss / count)
    #             },
    #             step=current_iteration,
    #         )
    print(
        "Validation: {}, validation loss: {}".format(
            current_iteration, loss / count
        ),
    )
    classifier.train()


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
    
