## SDF with NN

-------

### *The task is to simulate the SFD function for transforming the mesh 3D data format into the SFD.*

-------
### *Input data (mesh)*

Chair           |  Cube
:-------------------------:|:-------------------------:
![Alt text](images/chair.jpg?raw=true)  |  ![Alt text](images/cube.jpg?raw=true)

Pyramid           |  Teddy
:-------------------------:|:-------------------------:
![Alt text](images/pyramid.jpg?raw=true)  |  ![Alt text](images/teddy.jpg?raw=true)

-------

#### *I used fully-connected and convolutional neural networks to model SDF functions for input data.*
#### *Models:*
 - __Net7__ - NN with with seven fully-connected layers
 - __Net12__ - NN with with twelve fully-connected layers
 - __ConvNet__ - NN with with three convolutional layers
 - __ConvNet2__ - NN with with six convolutional layers

#### *Below comparison is the result of training four models for four input mesh figures.*

Cube           | SDF heat map           | Net7 output           | Net12 output          | CNN output          | CNN2 output          |
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|  
![Alt text](images/cube.jpg?raw=true)  | ![Alt text](images/cube_heatmap.jpg?raw=true)  | ![Alt text](images/cube_net7.jpg?raw=true)  | ![Alt text](images/cube_net12.jpg?raw=true)  |![Alt text](images/cube_cnn.jpg?raw=true)  |![Alt text](images/cube_cnn2.jpg?raw=true)  |  