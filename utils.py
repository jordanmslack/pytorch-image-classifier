import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn, tensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

import json
import argparse
from collections import OrderedDict


class Classifier(nn.Module):
    
    def __init__(self, input_features, hidden_units):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_units = hidden_units
        
        self.fc1 = nn.Linear(self.input_features, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, 256)
        self.fc3 = nn.Linear(256, 102)
        self.dropout = nn.Dropout(p=0.3)
            
    def forward(self, x):
        
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        
        return x
    
    
def model_architecture(network):
    
    arch = {"vgg16": [25088, models.vgg16(pretrained=True)], 
            "densenet121": [1024, models.densenet121(pretrained=True)], 
            "alexnet": [9216, models.alexnet(pretrained=True)]}
    
    if network not in ['vgg16', 'alexnet', "densenet121"]:
        print(f"""Im sorry! This application hasn't been configured to work with {network}. 
              Please try using either vgg16, densenet121 or alexnet.""")
    else:
        
        input_features = arch.get(network, '')[0]
        model = arch.get(network, '')[1]
        model.name = network
        
        for param in model.parameters():
            param.requires_grad = False 
    
        return model, input_features

    
def detect_device(gpu):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
        
    return device


def transform_image(data_dir):

    data_transforms = {
    'train_transforms':transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test_transforms' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'validation_transforms' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    
    image_datasets = {
        'train_data' : datasets.ImageFolder(f'{data_dir}/train', transform=data_transforms['train_transforms']),
        'test_data' : datasets.ImageFolder(f'{data_dir}/valid', transform=data_transforms['test_transforms']),
        'valid_data' : datasets.ImageFolder(f'{data_dir}/test', transform=data_transforms['validation_transforms'])
    }
    
    return image_datasets['train_data'] , image_datasets['valid_data'], image_datasets['test_data']


def load_data(data_dir):
    
    training_data, validation_data, test_data = transform_image(data_dir)
    
    dataloaders = {
        'trainloader' : torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True),
        'validloader' : torch.utils.data.DataLoader(validation_data, batch_size=32),
        'testloader' : torch.utils.data.DataLoader(test_data, batch_size=32),
        }
    
    return dataloaders['trainloader'] , dataloaders['validloader'], dataloaders['testloader']


def validation(model, validloader, criterion, device):
    
    loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(validloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return loss, accuracy


def train_network(model, trainloader, validloader, device, optimizer, epochs):
    
    steps = 0
    print_every = 10
    model.to(device)
    
    criterion = nn.NLLLoss()
    
    train_losses, validation_losses = [], []

    for e in range(epochs):
        
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            print(steps)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion, device)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(validloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(validloader)))
            
                running_loss = 0
                model.train()
            
            
def save_checkpoint(model, train_data, save_dir):

    model.class_to_idx = train_data.class_to_idx
    
    torch.save({'architecture': model.name,
                'class_to_idx': model.class_to_idx,
                'classifier': model.classifier,
                'state_dict': model.state_dict()}, 
                f'{save_dir}/checkpoint.pth')


def load_checkpoint(path):
    
    checkpoint = torch.load(path)
    
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else: 
        exec(f"model = models.{checkpoint['architecture']}(pretrained=True)")
        model.name = checkpoint['architecture']
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    model.classifier = checkpoint ['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    
    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    img = transform(image).unsqueeze_(0).float()
    
    return img


def predict(image_path, model, device, cat_to_name, top_k):

    model.to(device)
    img = process_image(image_path)
    
    with torch.no_grad():
        output = model.forward(img.to(device))

    probability = F.softmax(output.data, dim=1)
    
    return probability.topk(top_k)

    
def arg_parser(type=None):
    
    parser = argparse.ArgumentParser()
    
    if type == 'test':
        
        parser.add_argument('--arch', type=str, default='vgg16')
        parser.add_argument('--save_dir', type=str, default='/home/workspace/ImageClassifier')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--hidden_units', type=int)
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--gpu', action="store_true")
        
        return parser.parse_args()

    else:

        parser.add_argument('--image', type=str, default='/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg')
        parser.add_argument('--checkpoint', type=str, default='/home/workspace/ImageClassifier/checkpoint.pth')
        parser.add_argument('--top_k', type=int, default=5)
        parser.add_argument('--category_names', type=str, default='cat_to_name.json')
        parser.add_argument('--gpu', action="store_true")

        return parser.parse_args()
