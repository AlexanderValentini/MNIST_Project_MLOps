import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from model_cnn import MyAwesomeModel
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import src.data.data
wandb.init(project = "test-project", entity = "mlogs23")
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

# Is currently a




train_data_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/Cookie_Cutter_project_git_supported (newest)/MNIST_Project_MLOps/data/processed/train.pt"

# train_path = data_path + "train_set_processed_Mnist_corrupted.npy"
# test_path = data_path + "test_set_processed_Mnist_corrupted.npy"


@click.group()
def cli():
    pass


@click.command()
@click.argument("train_data_filepath", type=click.Path(exists=True))
@click.argument("lr", default=1e-3)
def train(train_data_filepath, lr):

#    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, ) as prof:
        model = MyAwesomeModel(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_dataloader = DataLoader(
            src.data.data.mnist(train_data_filepath), batch_size=16, shuffle=False
        )

        epochs = 5
        epochs_list = np.arange(0, epochs)
        train_losses = []
        for e in tqdm(range(epochs)):
            running_loss = 0
            for sample in train_dataloader:

                images = sample["image"]
                #        print(images.shape)
                labels = sample["labels"]
                optimizer.zero_grad()
                output_logits = model(images)
                                #           print(output_logits)
                loss = criterion(output_logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print("Training loss is", running_loss)
            train_losses.append(running_loss)
            wandb.log({"train_loss": running_loss}, step = e)
            print(type(running_loss))

        """ checkpoint = {'input_size': 784,
                    'output_size': 10,
                    'hidden_layers': [512, 384, 256, 128, 64, 32],
                    'drop_p': 0.2,
                    'state_dict': model.state_dict()}
        """

        torch.save(
            model.state_dict(),
            "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/models/MNIST_CNN_model_weights.pth",
        )

        print(train_losses)

        plt.plot(epochs_list, np.array(train_losses))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training loss as a function of epochs")
        plt.show()

    #prof.export_chrome_trace("trace.json")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

cli.add_command(train)

if __name__ == "__main__":

    cli()
