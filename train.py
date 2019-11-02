from utils import *

def main():
     
    args = arg_parser(type='test')
    
    gpu = args.gpu
    epochs = args.epochs
    learning_rate = args.learning_rate
    save_dir = args.save_dir
    
    device = detect_device(gpu=args.gpu)
    model, input_features = model_architecture(args.arch)
    
    if args.hidden_units:
        hidden_units = args.hidden_units
    else: 
        if input_features < 4096:
            hidden_units = 512
        else:    
            hidden_units = 4096
      
    print(f"We are using {model.name} as the selected model")
    print(f"Epochs set to: {epochs}")
    print(f"Learning Rate set to: {learning_rate}")
    print(f"Input features set to: {input_features}")
    print(f"Hidden units set to {hidden_units}")
    
    data_dir = 'flowers'
    
    train_data, valid_data, test_data = transform_image(data_dir)
    trainloader, validloader, testloader = load_data(data_dir)
    
    classifier = Classifier(input_features, hidden_units)
    
    model.classifier = classifier
    model.to(device)
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print("Preparing to train model")
    
    trained_model = train_network(model, trainloader, validloader, device, optimizer, epochs)
    save_checkpoint(model, train_data, save_dir)
    
    print("Checkpoint has been saved.")
    

if __name__ == '__main__': 
    main()