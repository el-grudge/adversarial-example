import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from ResNet import ResNet
from block import block

def generate_adversarial_example(image, label):
    img_dimensions = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize to the ImageNet mean and standard deviation
    # Could calculate it for the cats/dogs data set, but the ImageNet
    # values give acceptable results here.
    img_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_dimensions, img_dimensions)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
        ])

    # Prepare for attack
    image = Image.open(image)
    image = img_transform(image).float().unsqueeze(0)
    label = torch.tensor([int(label)])
    model = ResNet(block, [2, 2, 2, 2], 3, 2)
    model_path = './model_resnet18_epochs=5.pt'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    criterion = nn.CrossEntropyLoss()
    epsilon=.2

    image.requires_grad = True
    output = model(image.to(device))
    init_pred = torch.max(output.data, 1)[1] # save the initial prediction
    loss = criterion(output, label.to(device))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data

    # Perform attack: generate perturbed image
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # usually used with MNIST

    adv_output = model(perturbed_image.to(device))
    adv_pred = torch.max(adv_output.data, 1)[1] # save the prediction of the adversarial sample
    return [{'image': image, 'pred': init_pred}, {'image': perturbed_image, 'pred': adv_pred}]

def plot_image(image_dict):
    classes = ('dog', 'cat')
    plt.imshow(image_dict['image'].squeeze().detach().numpy().transpose((1,2,0)))
    plt.xticks([])
    plt.yticks([])
    plt.title(classes[image_dict['pred']])

def show_example():
     image = './data/cat.9957.jpg'
     label = 1
     natural, perturbed = generate_adversarial_example(image, label)
     plot_image(natural)
     plot_image(perturbed)