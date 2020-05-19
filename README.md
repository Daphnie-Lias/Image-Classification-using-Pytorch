# Udacity Nanodegree Image Classification Project
Create an Image Classfier

To run the files:
train.py :  python train.py data_dir --arch 'densenet' --learning_rate 0.01 --hidden_units 512 --epochs 3
If 'arch'parameter is omitted, default architecture 'vgg16' is considered.

predict.py : python predict.py --image "flowers/train/13/image_05765.jpg"  --top_k 3 

