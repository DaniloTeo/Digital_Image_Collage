# Digital_Image_Collage
## Students:
* Danilo da Costa Telles Téo - 9293626
* Rodrigo Valim Maciel - 9278149
* Victor Roberti Camolesi - 9791239

## Abstract:
### Objective:
The application uses image segmentation from inputted images for the purpose of creating collages. Hopefully getting aesthetically pleasing or at least interesting results. We'll use  a bigger image for definition of a background while smaller ones will be extracted of interesting features to be pasted on to the canvas.

### Datasets:
The images will be obtained from two different datasets:
* Corel1000 - chosen for the smaller images (.zip in repository);
* Flickr15K - chosen for the background images (http://kahlan.eps.surrey.ac.uk/data/Flickr25K/Flickr15K.zip)

Both datasets were suggested by professor Moacir Ponti.

### Steps:
The prefered approach to the project will be:
* Apply the OpenCV function, saliency(), on the smaller images in order to obtain a gray-scale image of the objects of interest;
* Apply another OpenCV function, threshold(), again on the smaller images to obtain a binary image which makes it easier to cut out the objects acquired in the first step;
* Next we find the contour of the objects in the image and actually cut it;
* The background will be randomly selected from the Flickr15K dataset;
* Paste the cut features on the background avoiding excessive overlap.

## Image Examples:
* see file example.png for example generated from the project's code
* https://br.pinterest.com/pin/47780446018928459/
* https://br.pinterest.com/pin/62276407317062641/
* https://br.pinterest.com/pin/552394710543721066/
