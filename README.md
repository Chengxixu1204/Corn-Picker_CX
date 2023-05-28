# Corn-Picker
The network(based on FCN) project is able to identify the multiple corns in a picture and at the same time figure out the relative distances towards the camera
Corn-Picker project could identify mulyiple corns and rank their importance in a picture
In oder to finish this purpose I adopt the stategy of belief map and FCN（https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf）
The first step: mark the belief map of images(./code/create_map)
The second step: train the network with marked images(./code/FCN_corn)
The third step: test the network(./code/valid_the_model)
All the data is located in(./data)
All the test results are located in(./result)
If you want to run the program, you could change the path accroding to your own needs
