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

Cube Mesh                  |                            Pyramid Mesh               |                                Teddy Mesh                 |                            Chair Mesh                 | 
:-------------------------:|                            :-------------------------:|                                :-------------------------:|                            :-------------------------:| 
![Alt text](images/cube.jpg?raw=true)  |                ![Alt text](images/pyramid.jpg?raw=true)  |                 ![Alt text](images/teddy.jpg?raw=true)  |               ![Alt text](images/chair.jpg?raw=true)  | 

SDF heat map for Cube      |                            SDF heat map for Pyramid   |                                SDF heat map for Teddy     |                            SDF heat map for Chair     | 
:-------------------------:|                            :-------------------------:|                                :-------------------------:|                            :-------------------------:| 
![Alt text](images/cube_heatmap.jpg?raw=true)  |        ![Alt text](images/pyramid_heatmap.jpg?raw=true)  |         ![Alt text](images/teddy_heatmap.jpg?raw=true)  |       ![Alt text](images/chair_heatmap.jpg?raw=true)  | 

Net7 output for Cube       |                            Net7 output for Pyramid    |                                Net7 output for Teddy      |                            Net7 output for Chair      |
:-------------------------:|                            :-------------------------:|                                :-------------------------:|                            :-------------------------:|
![Alt text](images/cube_net7.jpg?raw=true)  |           ![Alt text](images/pyramid_net7.jpg?raw=true)  |            ![Alt text](images/teddy_net7.jpg?raw=true)  |          ![Alt text](images/chair_net7.jpg?raw=true)  |

Net12 output for Cube      |                            Net12 output for Pyramid   |                                Net12 output for Teddy     |                            Net12 output for Chair     |  
:-------------------------:|                            :-------------------------:|                                :-------------------------:|                            :-------------------------:|  
![Alt text](images/cube_net12.jpg?raw=true)  |          ![Alt text](images/pyramid_net12.jpg?raw=true)  |           ![Alt text](images/teddy_net12.jpg?raw=true)  |         ![Alt text](images/chair_net12.jpg?raw=true)  |  

CNN output for Cube        |                            CNN output for Pyramid     |                                CNN output for Teddy       |                            CNN output for Chair       | 
:-------------------------:|                            :-------------------------:|                                :-------------------------:|                            :-------------------------:| 
![Alt text](images/cube_cnn.jpg?raw=true)  |            ![Alt text](images/pyramid_cnn.jpg?raw=true)  |             ![Alt text](images/teddy_cnn.jpg?raw=true)  |           ![Alt text](images/chair_cnn.jpg?raw=true)  | 

CNN2 output for Cube       |                            CNN2 output for Pyramid    |                                CNN2 output for Teddy      |                            CNN2 output for Chair      | 
:-------------------------:|                            :-------------------------:|                                :-------------------------:|                            :-------------------------:| 
![Alt text](images/cube_cnn2.jpg?raw=true)  |           ![Alt text](images/pyramid_cnn2.jpg?raw=true)  |            ![Alt text](images/teddy_cnn2.jpg?raw=true)  |          ![Alt text](images/chair_cnn2.jpg?raw=true)  | 

--------------------------------------------------------------------------------------------------------------

### *Conclusions:*
 - Even the simplest fully-connected models with a small number of layers are able to simulate SFD functions.
 - One-dimensional convolution is an effective method for this task. Even without normalization, 
 it describes the mesh well, and the weight of the model is less than the mesh file.
 - After a large number of experiments, such models will describe mesh objects well using less resources.
 - This is a great area for research ;)
--------------------------------------------------------------------------------------------------------------

### *TODO:*
 - Improve loss func (e.g. from DeepSDF)
 - Improve visualization
 - Parallelize inference models
 - More convolution network experiments
 - Add nn.BatchNorm1d
 - Use a model for group mesh objects
 - Etc

--------------------------------------------------------------------------------------------------------------
