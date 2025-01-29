# Inference with `inference.py`
The file `inference.py` accepts the path to the directory with two subdirectories ("aligned" and "not_aligned") of images. It then runs the image classification model on all images in the provided paths, makes predictions for them and prints the performance metrics: accuracy and F-score in percent points.
It also reports the averate latency of predictions.
For more information, run `python3 inference.py --help`.

# Presentation
[Link](https://www.loom.com/share/67f59543b96844a09bbb10e8fdfedf27?sid=24805bf0-3ba2-4a88-9283-2c5c0f9083e0)

# Installation

Install python virtual environment.

`pip install -r requirements.txt`

`pip install -e .`

For mac OS users (for labeling script):

`brew install python-tk`

# Labeling
Change `image_folder`, `local_prefix`, `server_prefix` in `src/labeling/labeling.py` for labeling.

`labeling.py` will produce two files:
- `class_align.txt`
- `class_not_align.txt`

in the `src/labeling` folder.

# Symlinks
Symlinks used for labeled images to avoid copying. 
Call `make symlinks` after getting the labeled lists with paths.

To remove symlinks, call `make clean`.


# Training
- For training the models, run the corresponding bash files:  
- For EfficientNetV2: `bash main_efficientnet.sh`  
- For Swin Transformer: `bash main_swin.sh`  
- For ConvNeXt: `bash main_convnext.sh`
