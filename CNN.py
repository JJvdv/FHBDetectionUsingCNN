import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from datetime import date
from NeuralNetwork import NeuralNetwork, torch, nn
from torchvision.transforms import transforms
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#################################################
# Root, model, and evaluation data directories
#################################################
ROOT_DIR = '.\\Data\\kaggle\\'
MODEL_DIR = '.\\models\\'
EVAL_DIR = '.\\Data\\kaggle\\evaluation\\'
SUB_DIR = '.\\submissions\\'
        
            
###########################################################################################################################
# class LoadDataset()
# Used to create TensorDatasets.
# CreateDataset(data_files, labels) - Creates TensorDataset for Training, Testing, and Evaluation data to train the model
###########################################################################################################################
class LoadDataset():
    def __init__(self):
        self.data_files = []
        self.labels = []        

    def CreateDataset(self, data_files, labels):
        transform = transforms.Compose([
            transforms.RandomRotation(degrees = (0, 180)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ])
    
        for i in range(len(data_files)):
            data_pt = torch.load(data_files[i], weights_only = False)
            
            self.data_files.append(data_pt)
            self.labels.append(torch.tensor(labels[i], dtype = torch.long)) 
            
            transformed_pt = transform(data_pt)
            self.data_files.append(transformed_pt)
            self.labels.append(torch.tensor(labels[i], dtype = torch.long))
            
            
        data_tensor = torch.stack(self.data_files)
        label_tensor = torch.stack(self.labels)

        dataset = TensorDataset(data_tensor, label_tensor)
        return dataset

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, i):
        data = self.data_files[i]
        label = self.labels[i]
        
        return data, label
    
#######################################################
# SaveModel(model)
# Used to save pytorch model as a pytorch tensor
######################################################
def SaveModel(model):
    curr_date = date.today()
    print('Saving model....')
    torch.save(model, MODEL_DIR + f'CNN-{curr_date}-Adam-CrossEntropy-Sigmoid.pt')
    print('Saviang model Completed...')
    
#########################################################
# Train(Epochs, model, dataloader_train, dataloader_test, opt)
# The train function to train the model
#########################################################
def Train(epochs, model, dataloader_train, dataloader_test, opt, device, criterion):
    print('Training Started.............')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        model.to(device)

        for inputs, targets in dataloader_train:
            inputs, targets = inputs.to(device), targets.to(device)

            opt.zero_grad()
            outputs = model(inputs).squeeze()
            targets = targets.float()
            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_accuracy = 100 * correct / total
        train_losses.append(train_loss / len(dataloader_train))
        
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in dataloader_test:
                inputs_val, targets_val = inputs.to(device), targets.to(device)
                outputs_val = model(inputs_val).squeeze()
                targets_val = targets_val.float()
                loss = criterion(outputs_val, targets_val)
                val_loss += loss.item()

                predicted_val = (outputs_val > 0.5).float()
                total_val += targets_val.size(0)
                correct_val += (predicted_val == targets_val).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_losses.append(val_loss / len(dataloader_test))
        
        print(f"EPOCH: {epoch + 1} / {epochs} -- train loss: {train_loss / len(dataloader_train):.4f} train accuracy: {train_accuracy:.4f} val loss: {val_loss / len(dataloader_test):.4f} val accuracy: {val_accuracy:.4f}")
    
    print('Training Completed..........')

    # Display model performance
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

    
#################################################################
# Load Data into 5 Lists (data, labels, evaluation)
#################################################################
print('Loading Data Started.............')
data_1_list = []
label_1_list = []
data_2_list = []
label_2_list = []
data_eval_list = []

for folder_name in os.listdir(ROOT_DIR):
    #print(folder_name) - debugging
    folder_path = os.path.join(ROOT_DIR, folder_name)
    #print(folder_path) - debugging
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if folder_name == '1':
            data_1_list.append(file_path)
            label_1_list.append(0)
        if folder_name == '2':
            data_2_list.append(file_path)
            label_2_list.append(1)
        if folder_name == 'evaluation':
            data_eval_list.append(file_path)

# Load the lists into numpy arrays so that it can be stacked
# Used so that it can be a concatenation of data 1 & 2 and labels 1 & 2
data_1 = np.array(data_1_list)
label_1 = np.array(label_1_list)
data_2 = np.array(data_2_list)
label_2 = np.array(label_2_list)
data = np.hstack((data_1, data_2))
labels = np.hstack((label_1, label_2))

# Using sklearn's train_test_split to split the lists of data into datasets.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2)
    
# Create variables that will hold the data in TensorDatasets for the CNN
train_data = LoadDataset().CreateDataset(X_train, y_train)
val_data = LoadDataset().CreateDataset(X_test, y_test)

# Loads the train_data & val_data TensorDatasets into DataLoaders from pytorch for model training.
# This will be what is pasted to the model and trained on.
dataloader_train = DataLoader(train_data, batch_size = 8, shuffle = True)
dataloader_test = DataLoader(val_data, batch_size = 8, shuffle = True)
print('Loading Data Completed..............')

####################################
# CNN Model Training
###################################
model = NeuralNetwork(num_channels = 101)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 1e-4
OPTIMIZER = optim.Adam(model.parameters(), lr = LR)
EPOCHS = 50
CRITERION = nn.BCELoss()


Train(EPOCHS, model, dataloader_train, dataloader_test, OPTIMIZER, DEVICE, CRITERION)
SaveModel(model)

################################################
# EVALUTION
# Loading evaluation data into DataLoader
################################################
eval_data = []

for file in data_eval_list:
    pt_data = torch.load(file, weights_only = False)
    eval_data.append(pt_data)
    
X_test = DataLoader(eval_data, shuffle = False)


######################################################
# Uncomment below to display predicted values
######################################################

print('Predicting validation set......')
all_predictions = []
model.to(DEVICE)
model.eval()

with torch.no_grad():
    for inputs in X_test:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs).squeeze()
        predicted = (outputs > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy().flatten())

print('All validation set predictions completed.....')        
print("Predictions:", all_predictions)


############################################################
# Uncomment below to create CSV and test accuracy on Kaggle
############################################################
'''
pred_list = []
curr_date = date.today()
for pred in all_predictions:
    if pred == 0:
        pred_list.append(1)
    if pred == 1:
        pred_list.append(2)
    
eval_id_files = []
for file in os.listdir(EVAL_DIR):
    eval_id_files.append(file)

file_path = os.path.join(SUB_DIR, f'Agri_submission-BCE-{curr_date}-pc.csv')
sub_df = pd.DataFrame({'Id': eval_id_files, 'Category': pred_list})
sub_df.to_csv(file_path, index = False)
'''