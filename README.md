# Annotation Tool
Annotation tool for labeling topological defects in images or videos

## Getting started
Prerequisites:

Install pyqt5 using pip
```
pip3 install PyQt5
```
Install numpy using pip
```
pip3 install numpy
```
Install scipy using pip
```
pip3 install scipy
```
Move all your images to ```Frames/images``` or videos to ```Videos/```

To label images, run ```python3 label_images.py```

Note that file names of images should be indices starting with 1 with zero paddings up to 6 digits.

To label videos, run ```python3 label_videos.py```

## Features
### Frame number selection
<kbd> CTRL </kbd>+<kbd> F </kbd> Then input frame number

<kbd> A </kbd> Or <kbd> &leftarrow; </kbd> to navigate to the previous frame

<kbd> D </kbd> Or <kbd> &rightarrow; </kbd> to navigate to the next frame

