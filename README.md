# NEURAL-STYLE-TRANSFER

COMPANY : CODTECH IT SOLUTIONS

NAME : HARANI C

INTERN ID : CT04DF167

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

# DESCRIPTION OF TASK 3 - NEURAL STYLE TRANSFER #
The objective of this task was to implement a Neural Style Transfer (NST) system using pre-trained models and deep learning libraries such as PyTorch and Torchvision. Neural Style Transfer is a fascinating computer vision technique that merges the content of one image with the style of another, creating a new image that appears to be painted in the style of the second image.

For this task, I built a complete Python script that accepts a content image and a style image, extracts features using a pre-trained VGG19 convolutional neural network, and then iteratively generates a stylized image by optimizing a target image. The final result is a beautiful image that retains the structure of the content image but has the textures, colors, and patterns of the style image.


# Tools and Libraries Used:
1.Python – Programming language

2.PyTorch – for deep learning framework and pretrained model (VGG19)

3.Torchvision – for transforms and loading VGG model

4.PIL (Python Imaging Library) – for image processing

5.torch.optim – for optimization using Adam

6.Matplotlib / Image.show() – For image visualization

7.Google Colab / Jupyter Notebook /VS Code – for implementation (depending on user)

 # How It Works:
1.Image Preprocessing:
The system begins by loading the content and style images using PIL and converting them into tensors using torchvision transforms. The images are resized, normalized based on ImageNet statistics, and converted into a 4D format required for PyTorch models.

2.Feature Extraction with VGG19:
The pre-trained VGG19 model, available in torchvision.models, is used to extract features from both content and style images. Only specific layers (like relu4_2 for content and relu1_1, relu2_1, etc., for style) are used based on prior research that shows these layers best capture the necessary information.

3.Gram Matrix for Style Representation:
To compare the style between two images, we calculate the Gram matrix of the feature maps. This matrix captures the spatial correlation between different filter responses and is used to represent style.

4.Loss Calculation:

Content Loss: Measures the difference between the content features of the original image and the target image.

Style Loss: Measures the difference between the Gram matrices of the style image and the target image for each selected layer.
The total loss is a weighted sum of these two.

5.Optimization Loop:
The target image is initialized as a clone of the content image and updated using Adam optimizer over 300 iterations. The optimizer minimizes the total loss by adjusting the pixel values of the target image.

6.Output Generation:
After the training loop, the stylized image is converted back into a displayable format, denormalized, and saved as output.jpg.

# 📁 Project Structure:
css
Copy
Edit
Task3_Neural_Style_Transfer/
├── neural_style_transfer.py
├── content.jpg
├── style.jpg
├── output.jpg

# 🎯 Outcome:
The final result is an image that preserves the objects, layout, and structure of the original content image while beautifully incorporating the color schemes, textures, and brush strokes of the style image. This task demonstrates the creative power of AI in visual arts and the practical use of transfer learning in deep learning.

# Conclusion:
This task was an excellent opportunity to explore both the theoretical and practical aspects of neural networks, feature extraction, and artistic AI applications. It reinforced my understanding of transfer learning, optimization, and image manipulation. Neural Style Transfer has real-world applications in the field of design, photo editing, content creation, and even in video stylization.

By completing this task, I have successfully implemented a working example of an NST model using VGG19, and I am now more confident in working with computer vision projects involving PyTorch and image-based deep learning models.

 # OUTPUT #

 ![Image](https://github.com/user-attachments/assets/21a409dc-f8a0-4d86-9d92-38b915f871d4)
