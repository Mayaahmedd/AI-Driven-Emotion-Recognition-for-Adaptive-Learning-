# AI-Driven-Emotion-Recognition-for-Adaptive-Learning-
bachelor thesis 2026 


3) pip install torch torchvision opencv-python matplotlib scikit-learn pandas jupyter

This installs the Python libraries you need.

torch → PyTorch, the main deep learning library  the main deep learning framework.
You need it to:

build the FER model
train the model
run inference
save/load checkpoints
______________________________________________________________________________________________________
torchvision → image tools and pretrained vision models like ResNet
This supports computer vision tasks in PyTorch.
You need it for:

pretrained image models like ResNet-50
image transforms
dataset/image utilities
______________________________________________________________________________________________________
opencv-python → image and webcam processing

This is OpenCV.
You need it for:

reading images
webcam capture
frame processing
possible face image handling during inference
______________________________________________________________________________________________________
matplotlib → plotting graphs
This is for plotting and visualization.
You need it for:

training curves
loss/accuracy graphs
confusion matrix plots
thesis figures if needed
______________________________________________________________________________________________________
scikit-learn → metrics like accuracy, F1-score, confusion matrix
This is mainly for evaluation utilities.
You need it for:

accuracy
precision
recall
F1-score
confusion matrix
train/validation utilities if needed
______________________________________________________________________________________________________
pandas → handling CSV/data tables
This is for table/CSV handling.
You need it for:

reading label files
managing dataset metadata
logging experiment results
______________________________________________________________________________________________________
jupyter → notebooks for experiments

This gives you Jupyter Notebook support.
You need it for:

quick experiments
debugging
testing preprocessing
visualizing sample predictions
______________________________________________________________________________________________________
facenet-pytorch
You installed this mainly for MTCNN.
You need it for:

face detection
face alignment
preparing webcam frames before FER classification
______________________________________________________________________________________________________
timm
This is a model library with many pretrained vision models.
You need it for:

easy access to models like ResNet, ViT, and others
faster experimentation without manually rebuilding models
______________________________________________________________________________________________________
tqdm
This is for progress bars.
You need it for:

cleaner training loops
seeing epoch/batch progress
better monitoring during long runs
-------------------------------------------------------------------------------------------------------
These files

#transforms.py
Contains preprocessing steps.

#model.py
Contains the model itself.

#train.py
This is the training script.

It will:

load the data
train the model
calculate loss
save the best checkpoint

#infer_image.py

Runs prediction on one image.

Good for testing:

“Does my model work on this face image?”

#infer_webcam.py

Runs prediction from the webcam in real time.

This comes later, not first.

#smoothing.py

Used later for temporal smoothing.

eg:
instead of trusting one frame
average the last few predictions
output a more stable emotion

#utils.py

Helper functions.
