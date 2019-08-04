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

### Labeling
<hr/>

- Repeat util finish labeling:

  - 1. Select frame
  
  - 2. Select annotation mode (Update, Append, Delete)
  
  - 3. Label desired points
  
  - 4. Save changes / Discard changes
  
- Save results to file

<hr/>

#### Frame selection
<kbd> ⌘ Command </kbd> + <kbd> F </kbd> Then input frame number

<kbd> A </kbd> or <kbd> &leftarrow; </kbd> to navigate to the previous frame

<kbd> D </kbd> or <kbd> &rightarrow; </kbd> to navigate to the next frame

<hr/>

#### Update Mode
In this mode, new annotations to one class will replace all existing annotations of that class.

<kbd>⇧ Shift</kbd> + class index (E.g. 1,2,3,4)

<hr/>

#### Append Mode
In this mode, new annotations to one class will be added to all existing annotations of that class.

<kbd>⌘ Command</kbd> + class index (E.g. 1,2,3,4)

<hr/>

#### Delete Mode
In this mode, existing annotations near selected point will be deleted.

<kbd>tab</kbd>

<hr/>

#### Save changes for current frame or discard changes
<kbd>Space</kbd> will save changes. If you don't want to save current changes, just begin selecting the next frame

<hr/>

#### Save annotations to file
<kbd>⌘ Command</kbd> + <kbd> S </kbd>
