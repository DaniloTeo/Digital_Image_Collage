# Digital\_Image_Collage

## Students:
* Danilo da Costa Telles Téo - 9293626
* Rodrigo Valim Maciel - 9278149
* Victor Roberti Camolesi - 9791239

## Report:
### Objective:
The application uses image segmentation from prepared datasets for the purpose of creating collages. Hopefully getting aesthetically pleasing or at least interesting results. We'll use  a bigger image for definition of a background while smaller ones will be extracted of interesting features to be pasted on to the canvas.

### Datasets:
The images will be obtained from two different datasets:
* Corel1000 - chosen for the smaller images (.zip in repository);
* Flickr15K - chosen for the background images (http://kahlan.eps.surrey.ac.uk/data/Flickr25K/Flickr15K.zip)

Both datasets were suggested by professor Moacir Ponti.

Certain images from each dataset were selected to better suit our project. We found the images which worked better for our 'cutting' method which are registered in the `test\_log.txt` file. These files form our sticker database which is located at the `Imagens_Teste` folder.

Another selection was made for the Flickr15K dataset, this time preferring landscape pictures rather than pictures of specific items such as sunflowers. The new dataset is located at the `My_Flickr` folder.

### Steps:
* The number of clusters for the K-Means algorithm is provided as input.

* The background will be randomly selected from the `My_Flickr` dataset;

* The next step is to obtain the sticker files from the `Imagens_Teste` dataset. For this, we attain to the following steps

* Apply the OpenCV function, saliency(), on the smaller images in order to obtain a gray-scale image of the objects of interest;

* Apply another OpenCV function, threshold(), again on the smaller images to obtain a binary image which makes it easier to cut out the objects acquired in the first step;

* Next we find the contour of the objects in the image and actually cut it;

* Then we calculate the mode for each of the RGB channels of every sticker file, in order to use these numbers for comparison in the K-Means algorithm;

* After that comes the actual use of K-Means aiming to classify the stickers in separate clusters and then use only the cluster in which the values are closer to the background's mode;

* Now the processing of the data is ready and we actually paste the stickers onto the background; 

* That is performed by selecting one of four regions on the background (based on the iteration number) on which to paste the images, aiming to avoid excessive overlap.


### Results:
This was an instigating project to work on since it provided the group with the rare opportunity to use computer code and algorithms in order to compose art. The results were all interesting and as we progressed we added functionalities mainly to improve the code's performance but also to improve the aesthetic of the output image.

We considered using the K-Means method yet again within the project for defining the sticker's paste coordinates on the background, but unfortunately said task would demand more time from us than we actually had available.

### Output results:
 See files in the `examples` folder

### Instructions for running the demo:
* On a terminal, inside de `demo` folder type in:
`python3 app.py`

* The program will request the number of clusters to be used in the K-Means algorithm. Keep in mind that the bigger this number, the less stickers will be pasted on the final output.

* To exit the program simply press '0', without closing the output image window.

## Image Examples:
* see file example.png for example generated from the project's code
* https://br.pinterest.com/pin/47780446018928459/
* https://br.pinterest.com/pin/62276407317062641/
* https://br.pinterest.com/pin/552394710543721066/
