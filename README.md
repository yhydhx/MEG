# MEG
Here is the code for MEG project.

Mainly this project contains two parts. 

First is the feature extraction part. We mainly use the DSRAE (Simultaneous Spatial-temporal Decomposition of
Connectome-scale Brain Networks by Deep Sparse Recurrent Auto-encoders). 

Second, we use the Randomly Wired Neural Networks (Exploring Randomly Wired Neural Networks for Image Recognition arXiv:1904.01569v2) for classify the 6-class problem. There is no offical code for this model. So I download the code from RandWireNN(https://github.com/seungwonpark/RandWireNN) and remove most of his code to simply it as much as possible.
The configuration are setted in the configs folder. So the second model is written by pytorch.


The graph models are saved in the model folder. 
You can set the number of blocks and the block size in Model_5block.py. Here I set the 5 blocks and for each block, it has 6 nodes.

