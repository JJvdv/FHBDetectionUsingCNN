''''''''''''''''''''''''''''''''''
' API endpoint for model predictions
'''''''''''''''''''''''''''''''''''
import uvicorn
import torch
import torch.nn as nn
import io
import base64

import numpy as np

from PIL import Image
from NeuralNetwork import NeuralNetwork
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torch.utils.data import DataLoader

app = FastAPI()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##########################################################################################################################
# predict() function:
# Request = POST
# input is a pt file that gets loaded through io stream
# the file gets converted to 3 channels to match input layer of ResNet50 Pre-trained Model
# The array that was converted to 3 channels gets laoded into a TensorDataset
# The TensorDataset gets loaded into a DataLoader so that the model can predict the file
# Predictions are done.
# 1 = Mild FHB.
# 2 = Serious FHB.
# The API POST request returns a dict of the model prediction on the file that was uploaded and the image of the file.
#######################################################################################################################
@app.post('/cnn/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        data_list = []
        file_content = await file.read()
        image_buffer = io.BytesIO(file_content)

        image_tensor = torch.load(image_buffer, weights_only = False)
        data_list.append(image_tensor)
        dataloader_pred = DataLoader(data_list)

        model = torch.load('./models/CNN-2024-10-15-Adam-CrossEntropy-Sigmoid.pth', weights_only = False)
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            for inputs in dataloader_pred:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                pred = (outputs > 0.5).float()
                pred = pred.cpu().numpy().flatten()
                if pred == 0:
                    response = 'Mild FHB'
                if pred == 1:
                    response = 'Serious FHB'
        
        image_buffer.close()
        
        tensor_3_channel = image_tensor[:3, :, :]
        tensor_transpose = np.transpose(tensor_3_channel, (1, 2, 0))
        tensor_norm = (tensor_transpose - tensor_transpose.min()) / (tensor_transpose.max() - tensor_transpose.min()) * 255
        np_arr = np.array(tensor_norm)
        
        img = np_arr.astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((200, 200), resample = Image.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format = 'PNG')
        buffer.seek(0)
        
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response_dict = {
        "prediction": response,
        "image": encoded_image
        }

        return JSONResponse(content = response_dict)
    except Exception as e:
        return str(e)
    
@app.post('/resnet/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        data_list = []
        file_content = await file.read()
        image_buffer = io.BytesIO(file_content)

        image_tensor = torch.load(image_buffer, weights_only = False)
        channel_transformer = nn.Conv2d(101, 3, kernel_size=1)
        channeled_data = channel_transformer(image_tensor)
        data_list.append(channeled_data)
        dataloader_pred = DataLoader(data_list)

        model = torch.load('./model20240807-Adam-8020.pt', weights_only = False)
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            for inputs in dataloader_pred:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                outputs = torch.sigmoid(outputs)
                pred = (outputs > 0.5).float()
                pred = pred.cpu().numpy().flatten()
                
            for p in pred:
                if p == 0:
                    response = 'Mild FHB'
                if p == 1:
                    response = 'Serious FHB'
        
        image_buffer.close()
        
        tensor_3_channel = image_tensor[:3, :, :]
        tensor_transpose = np.transpose(tensor_3_channel, (1, 2, 0))
        tensor_norm = (tensor_transpose - tensor_transpose.min()) / (tensor_transpose.max() - tensor_transpose.min()) * 255
        np_arr = np.array(tensor_norm)
        
        img = np_arr.astype(np.uint8)
        image = Image.fromarray(img)
        image = image.resize((200, 200), resample = Image.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format = 'PNG')
        buffer.seek(0)
        
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        response_dict = {
        "prediction": response,
        "image": encoded_image
        }

        return JSONResponse(content = response_dict)
    except Exception as e:
        return str(e)        

##################################################################################
# localhost = "xxx.xxx.xxx.xxx" - IP you wish to run the API on.
# port = xxxx - The port the API will run on.
# uviconr.run() - Runs the FastAPI app on the host address and port set below
##################################################################################
if __name__ == "__main__":
    localhost = "127.0.0.1"
    port = 8080
    uvicorn.run(app, host= localhost, port= port)