import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

data_path = "D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/Cookie_Cutter_project_git_supported (newest)/MNIST_Project_MLOps/data/processed/"

train_path = data_path + "train.pth"
#test_path = data_path + "test_set_processed_Mnist_corrupted.npy"


class mnist(Dataset):

    '''
    Wraps the selected dataset into a dataset class. Allows the dataloader to access the samples (image and label) one at a time as dictionaries. The length of each sample is the batch size.  

            Parameters:
                    filepath (str): Path to the processed dataset in .pt format (Either test or train)

            Returns:
                    sample (dict): A dictionary with an images and labels (of batchsize) is returned each time the getitem function is called. 
    '''

    def __init__(self, filepath, transform=None):

        self.samples = torch.load(filepath)

        self.images = self.samples["images"]
#        print(self.images.shape)
        self.labels = self.samples["labels"]

    def __len__(self):
        return self.images.shape[0]
    
    #We effectively split the large dictionary into small dictionaries each containing images and labels with the length of the batch size.   
    def __getitem__(self, index):

        sample = {"image": self.images[index], "labels": self.labels[index]}

      #  print(self.images[index].shape)

        return sample

#train_dataloader = DataLoader(mnist(train_path), batch_size = 16, shuffle = False)

""" for sample in train_dataloader: 
    image = sample['image']
    labels = sample['labels']
    print(image.shape)
    print(labels.shape)

    print(image[0].shape)
    print(image[0].permute(1, 2, 0).detach().numpy().shape)
    break 

sample = next(iter(train_dataloader))
image = sample['image']

plt.imshow(image[0].permute(1, 2, 0).detach().numpy())
plt.show()
 """