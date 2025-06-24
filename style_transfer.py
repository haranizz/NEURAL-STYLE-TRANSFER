import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import copy

# Load images
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

# Paths
content_path = "content.jpg"
style_path = "style.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load images
content = load_image(content_path).to(device)
style = load_image(style_path, shape=content.shape[-2:]).to(device)

# Define a model and loss functions (VGG19 for style transfer)
vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

vgg.to(device)

# Layers to use for content and style
content_layers = ['21']  # relu4_2
style_layers = ['0', '5', '10', '19', '28']  # relu1_1, relu2_1, relu3_1, relu4_1, relu5_1

def get_features(image, model, layers=None):
    features = {}
    x = image
    if layers is None:
        layers = content_layers + style_layers
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# Calculate gram matrices for style features
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create a target image and initialize it as a copy of the content image
target = content.clone().requires_grad_(True).to(device)

# Define weights for each style layer
style_weights = {'0': 1.0, '5': 0.75, '10': 0.2, '19': 0.2, '28': 0.2}
content_weight = 1e4
style_weight = 1e2

# Optimizer
optimizer = optim.Adam([target], lr=0.003)

# Style transfer loop
epochs = 300
for i in range(epochs):
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['21'] - content_features['21'])**2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if i % 50 == 0:
        print(f"Iteration {i}, Total loss: {total_loss.item()}")

# Save output image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    image = (image * 255).astype('uint8')
    return Image.fromarray(image)

output = im_convert(target)
output.save("output.jpg")
output.show()
