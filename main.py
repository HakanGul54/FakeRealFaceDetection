import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import model
import numpy as np
import matplotlib.pyplot as plt


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_model(model, data_loader, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}, actual: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Define the CNN Model
    classifier = model.Classifier(num_channels=3, input_size=300, output_size=2).to(device)

    # Load the data
    data = dset.ImageFolder(root="Data",
                            transform=transforms.Compose([
                                        transforms.Resize(300),
                                        transforms.CenterCrop(300),
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
        epoch_acc = (acc_data / len(data)) * 100
        print("total correct", acc_data, "total data length", len(data))
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

        visualize_model(classifier, data_loader, 10)
        plt.show()
