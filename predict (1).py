import argparse
from PIL import Image
import torch
import numpy as np
from train import check_gpu
from train import train_transform
from torch import nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import json

def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Neural Network Prediction")

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction. like "flowers/train/13/image_05765.jpg"',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=False)
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to labels')
    
    # Add GPU Option to parser
    parser.add_argument('--GPU', 
                        help='Option to use GPU', type = str)

    # Parse args
    args = parser.parse_args()
    
    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def load_checkpoint(checkpoint_path):
    
    if type(checkpoint_path) == type(None):
        checkpoint_path = "/home/workspace/ImageClassifier/checkpoint.pth"
    
    model = models.vgg16(pretrained=True)
    #Load Defaults if none specified
#     if checkpoint ['architecture'] == 'vgg16':
#          model = models.vgg16(pretrained=True)
#     else: #vgg13 as only 2 options available
#         model = models.vgg13 (pretrained = True) 
      
    checkpoint = torch.load (checkpoint_path,map_location=lambda storage, loc: storage) #loading checkpoint from a file
  
    
    model = models.vgg16(pretrained=True);
    model = nn.DataParallel(model)    
    model.classifier = checkpoint ['classifier']
    #model.load_state_dict (checkpoint ['state_dict'],strict=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint ['class_to_idx']
    
    for param in model.parameters(): 
        param.requires_grad = False #turning off tuning of the model
    
    return model

#Image processing

def process_image(image_path):
    test_image = Image.open(image_path)

    # Get original dimensions
    orig_width, orig_height = test_image.size

    # Find shorter size and create settings to crop shortest side to 256
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    # Find pixels to crop on to create 224x224 image
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    # Converrt to numpy - 244x244 image w/ 3 channels (RGB)
    np_image = np.array(test_image)/255 # Divided by 255 because imshow() expects integers (0:1)!!

    # Normalize each color channel
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    # Set the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model,cat_to_name,topk,device):
#     model.to("cpu")
    
    # Set model to evaluate
    model.eval();

    # The image
    image = process_image(image_path)
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    output = model.to(device).forward(image)
    
    print(image.shape)
    probabilities = torch.exp(output)
    print(probabilities)
    # getting the topk (=5) probabilites and indexes
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    print(prob)
    print(index)
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[index[i]])

    return prob, label

def get_image_classes(train_dir='flowers/train'):
    
    image_datasets_train = train_transform(train_dir)
    class_names = image_datasets_train.classes
#     print(class_names)
    return class_names


def main():
    """
    Executing relevant functions
    """
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Pre load categories to names json file
    if type(args.category_names) == type(None):
        category_names = 'cat_to_name.json'
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
     
    # Check for GPU
    device = check_gpu(gpu=args.GPU);
    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_index = predict(args.image, model.to(device),cat_to_name,args.top_k,device)
   
    print(top_probs)
    print(top_index)
    class_names = [cat_to_name [item] for item in top_index]
    print(flower_names)
    
    
   
# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': main()    
    

