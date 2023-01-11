# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torchvision import transforms

data_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/data/raw"

output_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/data/processed"



#transformation = transforms.Compose([
#    transforms.ToTensor()])                             
@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())

def main(input_filepath, output_filepath):

    '''
    The function processes the raw MNIST image data by normalizing it anc converting them into tensors. Then it saves the data as a dictionary. 
    Furthermore it returns the data in a dictionary. The dictionary contains the "images" and "labels" keys.

            Parameters:
                    input_filepath (str): Filepath to directory containing data
                    output_filepath (str): Path where the training and test files are stored as .pt dictionaries. 

            Returns:
                    train_data_dict (dict): Dictionary with the training data
                    test_data_dict (dict): Dictionary with the training data
    '''
    
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    data_path = input_filepath

    train_files, test_files = [], []

    #It goes through all files and folders in the directory:  
    for root, dirs, files in os.walk(data_path):
        #We save all the test and train datafiles seperately in two lists: 
        for file in files:
            if file[:5] == "train":
                train_files.append(np.load(os.path.join(root,file)))
            elif file[:4] == "test":
                test_files.append(np.load(os.path.join(root,file)))

    
    # The training images are extracted and concatenated into a [25000, 28, 28] numpy ndarray. 
    train_images = [f["images"] for f in train_files]
    train_images = np.concatenate(train_images)

    #We normalize the image datasets by using the mean and std of the training data. 
    train_mean = np.mean(np.mean(train_images, axis = 0))
    train_std = np.std(train_images)

    transform = transforms.Compose([transforms.Normalize(mean=train_mean,
                             std=train_std)])



    #Then they are converted to a pytorch tensor and a dummy "channel dimension" is added. This is in order to use the torchvision transforms normalization function.  
    train_images = torch.from_numpy(train_images)
    train_images = train_images.view(len(train_images),1, 28,28)

  
  # The training image dataset is normalized:
    transformed_train_images = transform(train_images)

  #The corresponding training labels are extracted, and concatenated into a [25000] ndarray
    train_labels = [f["labels"] for f in train_files]
    train_labels = np.concatenate(train_labels)
    train_labels = torch.from_numpy(train_labels)

    #The training dataset is saved as a dictionary. The labels and images are saved as tensors under seperate keys.  
    train_data_dict = {"images": transformed_train_images, "labels": train_labels}
  #  print(torch.mean(transformed_images))
  #  print(torch.std(transformed_images))
  #  print(transformed_images.shape)

# Extract test images and concatenate these into a [25000, 28, 28] numpy ndarray
    test_images = test_files[0]["images"]

#Conversion to pytorch tensor:
    test_images = torch.from_numpy(test_images)
    # Add the channel dimension. The resulting dimensions are (25000, 1, 28, 28)
    test_images = test_images.view(len(test_images),1, 28,28)
    #Normalization using training mean and std:
    transformed_test_images = transform(test_images)

        
    # Extract test labels and concatenate these into a [25000,] numpy ndarray
    test_labels = test_files[0]["labels"]
    test_labels = torch.from_numpy(test_labels)
    #The test data is similarly saved in a dict: 
    test_data_dict = {"images": transformed_test_images, "labels": test_labels}

    #The dicts are then saved on the disk at the output filepath as .pt files
    torch.save(train_data_dict, os.path.join(output_filepath, "train.pt"))
    torch.save(test_data_dict, os.path.join(output_filepath, "test.pt"))



    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    return train_data_dict, test_data_dict

if __name__ == '__main__':



    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()