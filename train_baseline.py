import time
import torch
from tqdm import tqdm
from pathlib import Path
import copy
import os
import sys
import random
import numpy as np
from protbert import Baseline
from baselineDataset import MabMultiStrainBinding

from sklearn.model_selection import train_test_split

# Reproducibility
BDRSEED = 529
torch.manual_seed(BDRSEED)
random.seed(BDRSEED)
np.random.seed(BDRSEED)

# PARAMS
save_folder = sys.argv[1]
num_epochs = int(sys.argv[2])
batch_size = int(sys.argv[3])  # Max batch: 8
print(
    f"\n#Params\nsave_folder: {save_folder} -- num_epochs: {num_epochs} -- batch_size: {batch_size} -- seed: {BDRSEED}"
)
print()


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split
    )
    datasets = {}
    datasets["train"] = torch.utils.data.Subset(dataset, train_idx)
    datasets["test"] = torch.utils.data.Subset(dataset, val_idx)
    return datasets


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=1):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e6

    print("TRAINING:\n")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print()
        print("#" * 10)
        print(f"Epoch {epoch+1}/{num_epochs}")

        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # iteration over the data
            running_loss = 0.0
            # running_correct = 0

            print("#" * 5)
            for ii, inputs in enumerate(dataloaders[phase]):
                # inputs = inputs.to(device)
                seqs, labels = inputs
                labels = labels.float().to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(seqs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size
                isPrint = (
                    True if ii % 50 == 0 or ii == dataset_sizes[phase] - 1 else False
                )
                if isPrint:
                    print(
                        f"{phase} {ii}/{dataset_sizes[phase]} Loss: {loss:.4f} Running Loss: {running_loss:.4f}"
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"\n{phase} Loss: {epoch_loss:.4f}")
            if phase == "test" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = Path(save_folder)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                checkpoint_path = os.path.join(
                    save_path, f"epoch-{epoch+1}_loss-{epoch_loss:.4f}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "batch_size": batch_size,
                    },
                    checkpoint_path,
                )

        time_elapsed_epoch = time.time() - epoch_start
        print(
            f"Epoch completed in {time_elapsed_epoch//3600:.0f}:{(time_elapsed_epoch//60)%60:.0f}:{time_elapsed_epoch%60:.0f}"
        )
        print("#" * 20)
    time_elapsed = time.time() - since
    print(
        f"Training completed in {time_elapsed//3600:.0f}:{(time_elapsed//60)%60:.0f}:{time_elapsed%60:.0f}"
    )

    model.load_state_dict(best_model_wts)

    return model


# Initialize

dataset = MabMultiStrainBinding(
    Path("/data2/dcardamone/deepcov/test/dataset.txt"), None
)
print("# Dataset created")

# train/test split

datasets = train_val_dataset(dataset=dataset)

train_loader = torch.utils.data.DataLoader(
    datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# next(iter(train_loader))

test_loader = torch.utils.data.DataLoader(
    datasets["test"],
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

dataloaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {
    "train": len(train_loader) * batch_size,
    "test": len(test_loader) * batch_size,
}

# Set the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device {0}".format(device))

# Select the model
model = Baseline(nn_classes=len(dataset.var_list), freeze_bert=True)
# model = model.to(device)

# Define criterion (must criterion = torch.nn.CrossEntropyLoss())
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

_ = train_model(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    dataloaders=dataloaders,
    num_epochs=num_epochs,
    dataset_sizes=dataset_sizes,
)


print("\n ## Training DONE ")


"""
# TESTING
python baselineTrain.py ./17oct22 10 8


nohup python baselineTrain.py ./18oct22-4classes_large_classifier 50 8 > 18oct22-4classes_large_classifier.out &
nohup python baselineTrain.py ./18oct22-4classes_dropout_1e-2rate 50 8 > 18oct22-4classes_dropout_1e-2rate.out &
"""