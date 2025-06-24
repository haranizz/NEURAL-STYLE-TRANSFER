# NEURAL-STYLE-TRANSFER

COMPANY : CODTECH IT SOLUTIONS

NAME : HARANI C

INTERN ID : CT04DF167

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 4 WEEKS

MENTOR : NEELA SANTHOSH KUMAR

# DESCRIPTION OF TASK 3 - NEURAL STYLE TRANSFER #
To build a functional Neural Style Transfer system that blends the content of one image with the artistic style of another using pre-trained deep learning models.

# Tools and Libraries Used:
1.PyTorch – for deep learning framework and pretrained model (VGG19)

2.Torchvision – for transforms and loading VGG model

3.PIL (Python Imaging Library) – for image processing

4.torch.optim – for optimization using Adam

5.Google Colab / Jupyter Notebook /VS Code – for implementation (depending on user)

 # How It Works:
1.Image Loading and Preprocessing:

The load_image() function loads and resizes the content and style images, then normalizes them using ImageNet statistics.

2.Feature Extraction with VGG19:

A pre-trained VGG19 model is used to extract features from specific layers (e.g., relu4_2 for content and multiple layers for style).

3.Style Representation with Gram Matrices:

Style features are converted into Gram matrices to capture texture and style patterns.

4.Optimization Loop:

A clone of the content image is used as the target.

The loss function combines content loss and style loss.

The target image is optimized using Adam optimizer over 300 iterations to match the style and content requirements.

5.Output:

The final stylized image is saved as output.jpg and displayed.

#Input Files:
content.jpg – The original content image

style.jpg – The image providing the artistic style

#Output:
output.jpg – The stylized image combining content and style

# Recommended File Name for Submission:
✅ If you're submitting a Python script:
neural_style_transfer.py

✅ If submitting as a Jupyter/Colab notebook:
Neural_Style_Transfer.ipynb

✅ Include sample images in your folder:

css
Copy
Edit
Task3_Neural_Style_Transfer/
├── neural_style_transfer.py
├── content.jpg
├── style.jpg
├── output.jpg

