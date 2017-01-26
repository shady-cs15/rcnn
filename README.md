# RCNN
Multiscale Recurrent Convolutional Neural Network for Scene Labeling <br>
<img src="https://raw.githubusercontent.com/shady-cs15/shady-cs15.github.io/master/images/rcnn.jpg"/> <br>
report at <a href="https://github.com/shady-cs15/shady-cs15.github.io/blob/master/files/rcnn_report.pdf"/>this link</a>

# Dependencies (Python)
1. Theano <br>
2. Matplotlib <br>
3. cPickle <br>
4. numpy <br>
5. pillow <br>

# Running
navigate to directory 'rcnn' <br>
make directory 'data' inside 'rcnn' <br>
put labeled images in data/labeled_scaled <br>
put rgb images in data/left_scaled <br>
navigate to 'src' directory <br>
```
python run.py
```
<br>Running on smaller data <br>
```
python test.py
``` 

# Running with GPU
Make sure proper driver and Nvidia CUDA toolkit is installed.
add the following lines to ~/.theanorc
```
[global]
device=gpu
floatX=float32
```
