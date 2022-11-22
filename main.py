import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import model

if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the CNN Model
    classifier = model.Classifier(num_channels=3, input_size=224, output_size=2).to(device)

    # Load the data
    data = dset.ImageFolder(root="Data",
                            transform=transforms.Compose([
                                        transforms.Resize(64),
                                        transforms.CenterCrop(64),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    data_loader = torch.utils.data.DataLoader(data, batch_size=14, shuffle=True, num_workers=2)
    class_names = data.classes
    
    # Set the Loss (Cross Entropy) and Optimizer (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 25)
        # Initialize Loss
        loss_data = 0
        acc_data = 0

        # Loop through data
        for i, (inputs, labels) in enumerate(data_loader):
            # Reset optimizer
            optimizer.zero_grad()

            # Convert data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward Pass
            outputs = classifier(inputs)
            _, predictions = torch.max(outputs, 1)
            
            # Backward Pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate Data
            loss_data += loss.item() * inputs.size(0)
            acc_data += torch.sum(predictions == labels.data)
        
        # Calculate and print epoch data
        epoch_loss = loss_data / len(data)
        epoch_acc = (acc_data.double() / len(data)) * 100
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
