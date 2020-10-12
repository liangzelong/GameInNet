<!--
 * @Author: langelo
 * @Date: 2020-09-30 14:20:02
 * @LastEditTime: 2020-10-12 13:38:43
 * @LastEditors: langelo
 * @Description: 
-->
# GameInNet
Embedding game in Neural Network
## Installation
##### Clone and install requirements
    $ git clone https://github.com/liangzelong/GameInNet.git
    $ cd GameInNet/
    $ sudo pip3 install -r requirements.txt
    $ python GameInNet.py

## Random Mouse Position Test

    $ python ./model/model.py

# MUSHROOM WILL CATCH YOU LITTLE SLIME，RUN!!!!!!
## Target Game
<p align="center"><img src="./image/org.gif" width="480"\></p>
This is our target game: slime escapes from mushroom.

## Generated Game
<p align="center"><img src="./image/gen.gif" width="480"\></p>
This is the image we generated.

## Random Mouse Test
<p align="center"><img src="./image/real_gan2.gif" width="480"\></p>
Test the program with a random nouse position. Slime will go towards mouse point and mushroom will follow slime. Mouse position is shown with red point.

## Generated Game
<p align="center"><img src="./image/real_gan3.gif" width="480"\></p>
Our game works well. Mouse position is shown with red point.

## Analysis the Movement of Entity
<p align="center"><img src="./image/data.gif" width="480"\></p>
This image shows the analysis of the relationship between mushroom's position and slime's position.