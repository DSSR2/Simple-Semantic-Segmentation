# Super Simplified Semantic Segmentation
## What is semantic segmentation and when do we need it?
It is the process of partitioning an image into multiple segments (sets of pixels, also known as image objects). The idea is to accuractely classify each and every pixel to one of the possible classes. 

When the task is to very accurately localise the object or defect of interest. While this is also a blackbox classifier and does not provide exact filters, it can build segmentation masks which exactly take on the shape of the object it is trying to locate. 

## Choice of models:
For ease of use, we have defined 6 models - S0, S1, S2, S3, S4, S5. 

S0 is the smallest and simplest model which will train the fastest but may not have very high accuracy. 
S5 is the largest and most complex. It will take some time to train but give a very high accuracy. 

If you have a simple dataset with uniform objects and less textures, start with S0 and move up if accuarcy is not satisfactory. If the dataset is complex, start with S4 and move up if accuracy is not satisfactory. 

## Folder structure to be followed:
For each image *.jpg, there must be a corresponding mask with the same filename in the mask folder. 
```
Root
    |--- Images
    |   |--- File1.jpg
    |   |--- File2.jpg
    |   |--- File3.jpg
    |   |--- ...
    |--- Masks
    |   |--- File1.png
    |   |--- File2.png
    |   |--- File3.png
    |   |--- ...
```
## Mask and Image details:
You can use any popular image annotation tool to annotate the image using the tool of your choice and convert from ```json``` or ```xml``` to ```.png``` segmentation maps.

The image and the mask have to have the same name so that the software can map the images and their maps correctly. The images and their corresponding masks are stored in the folder structure shown above. 

## Code Samples
The complete example code is provided in Example.py. 
