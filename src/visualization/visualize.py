import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.data
#from src.models.model_cnn import MyAwesomeModel
from src.models.model_cnn_reduced import MyAwesomeModel

model_weights_path = 'D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/models/MNIST_CNN_model_weights.pth'
#data_path = 
train_path = 'D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/data/processed/train.pt'


@click.group()
def cli():
    pass


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('data_filepath_processed', type=click.Path())

def visualize(model_path, data_filepath_processed):

    '''
    Makes a TSNE visualization of the feature space for the training dataset after it has been passed through three convolutional layers.

            Parameters:
                    model_path (str): Filepath for the pretrained model weights
                    data_filepath_processed (str): Path for the processed training data

            Returns:
                    Nothing (None): Visualization is saved directly to reports/figures
    '''

    #Load state dict and pass it to the model. 
    state_dict = torch.load(model_path)
    model = MyAwesomeModel(num_classes = 10)
    #Since a reduced CNN model without the final layer is used, the coresponding weights are removed from the state dict:  
    state_dict.pop('fc.bias', None)
    state_dict.pop('fc.weight', None)
    model.load_state_dict(state_dict)


    train_dataloader = DataLoader(src.data.data.mnist(filepath = data_filepath_processed), batch_size = 16, shuffle = False)
    #Initial value for total_feature_space
    total_feature_space = 0
    #Labels are loaded in for later TSNE visualization. 
    labels = torch.load(data_filepath_processed)['labels'].detach().numpy()
    with torch.no_grad():
        
        for sample in tqdm(train_dataloader):
            images = sample['image']
            feature_space = model(images)

            #For the first iteration the total_feature_space is set to be the current feature space for the batch [16,16,3,3]. 
            #Afterwards the feature spaces for each batch are iteratively concatenated. This gathers the features for the whole dataset. 
            if type(total_feature_space) == type(feature_space): 
                
                #This creates a tensor of size (25000,16,3,3)
                total_feature_space = torch.cat((total_feature_space, feature_space), dim = 0)
            else:
                total_feature_space = feature_space
        #Reshape feature space into (25000, 144) tensor
        total_feature_space_reshaped = total_feature_space.view(25000,-1).detach().numpy()
#            print(first_feature_space.shape)

    print(total_feature_space_reshaped.shape)
    #Runs the TSNE algorithm - reduces the dimensions to 2: 
    embedded_space = TSNE(n_components=2).fit_transform(total_feature_space_reshaped)

    #We plot all datapoints in the embedded space, while keeping track of their labels. Similarly the points are colored according to their label. 
    plt.scatter(embedded_space[:,0],embedded_space[:,1], s = 5, c = labels, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10)) 
    plt.title('Visualizing MNIST through t-SNE', fontsize=18)
    plt.savefig('D:/shared/DTU/7. Semester/Machine Learning Operations/Own Exercises folder/CookieCutter_project/Alex_Cookie_Cutter_repo/reports/figures/TSNE_visualization_of_MNIST.png')
#    plt.show()
    return

#    k = 2
    
cli.add_command(visualize)

if __name__ == '__main__':
    cli()
 
#visualize_tsne(model_path = model_weights_path, data_filepath_processed=train_path)