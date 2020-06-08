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
Install pandas using pip
```
pip3 install pandas
```

To label images, run ```python3 label_images.py```


## Labeling

- Repeat until finish labeling:

  - Select frame
  
  - Select annotation mode (Update, Append, Delete)
  
  - Label desired points
  
  - Save changes / Discard changes
  
- Save results to file


### Frame selection
<kbd> ⌘ Command </kbd> + <kbd> F </kbd> Then input frame number

<kbd> A </kbd> or <kbd> &leftarrow; </kbd> to navigate to the previous frame

<kbd> D </kbd> or <kbd> &rightarrow; </kbd> to navigate to the next frame

### Update Mode
In this mode, new annotations to one class will replace all existing annotations of that class.

<kbd>⇧ Shift</kbd> + class index (E.g. 1,2,3,4)

### Append Mode
In this mode, new annotations to one class will be added to all existing annotations of that class.

<kbd>⌘ Command</kbd> + class index (E.g. 1,2,3,4)

### Delete Mode
In this mode, existing annotations near selected point will be deleted.

<kbd>tab</kbd>

### Save changes for current frame or discard changes
<kbd>Space</kbd> will save changes. 

If you don't want to save current changes, just begin selecting the next frame

### Save annotations to file
<kbd>⌘ Command</kbd> + <kbd> S </kbd>

Note that the result will be saved in .mat format. To convert to DataFrame stored in .csv format, see the description below


## Author
Zhaonan Li, [LIZHAONAN](https://github.com/LIZHAONAN)

