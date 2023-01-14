import click
import numpy as np
import torch
import torch.nn.functional as F
from model_cnn import MyAwesomeModel
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import src.data.data
import src.data.make_dataset

model_weights_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/Cookie_Cutter_project_git_supported (newest)/MNIST_Project_MLOps/models/MNIST_CNN_model_weights.pth"

# test_path = data_path + "/test.npz"

# output_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/data/processed"

test_path_processed = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/data/processed/test.pt"


# TODO: Implement evaluation logic here


@click.group()
def cli():
    pass


# We initialize the click command and the arguments for the function.
@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data_filepath_processed", type=click.Path())
def evaluate(model_path, data_filepath_processed):

    """
    Evaluates the trained model on the MNIST test images. Shows the test loss and accuracy.

            Parameters:
                    model_path (str): Path to the pretrained model weights
                    data_filepath_processed (str): Filepath for the test data.

            Returns:
                    Nothing (None): All output information is printed.
    """

    # The model is initialized and weights are loaded in.
    state_dict = torch.load(model_path)
    model = MyAwesomeModel(num_classes=10)
    # Weights are applied to the model
    model.load_state_dict(state_dict)
    model.eval()
    # Test data is loaded. Uses the mnist dataset class.
    test_dataloader = DataLoader(
        src.data.data.mnist(filepath=data_filepath_processed),
        batch_size=16,
        shuffle=False,
    )
    print("Data Loaded Successfully")

    accuracies = []
    criterion = nn.CrossEntropyLoss()

    # We set the tensors to not save the gradients. This is in order to speed up computation.
    with torch.no_grad():

        running_loss = 0
        for sample in test_dataloader:
            images = sample["image"]
            labels = sample["labels"]
            output_logits = model(images)

            loss = criterion(output_logits, labels)
            # As the outputs are logits we convert it to probabilities:
            ps_val = F.softmax(output_logits, dim=1)

            # The predicted label is the one with the highest probability.
            top_p_val, top_class_val = ps_val.topk(1, dim=1)

            # We fine the number of correct predictions. This returns True/False boolean values.
            equals = top_class_val == labels.view(top_class_val.shape)

            # Then we calculate the accucaty as the mean value of the True/False predictions.
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            accuracies.append(accuracy)

            # The loss is added cumulatively
            running_loss += loss.item()

        # We output the test loss and accuracy
        print("Test loss is", running_loss)
        print(f"Accuracy: {np.mean(accuracies)*100}%")
    return


# Used to run the function from the command line
cli.add_command(evaluate)

# Runs click when the script is run
if __name__ == "__main__":
    cli()


# evaluate(model_path = model_weights_path, data_filepath_processed = test_path_processed,process_data = True)
