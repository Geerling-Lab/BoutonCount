# BoutonCount
This code is designed to both run and retrain BoutonCount

Running:

Tested using Python 3.8.11, numpy 1.20.3, scikit-image 0.18.1, pillow 8.3.1, keras 2.4.3, tensorflow 2.3.1. It was also tested using 20x EFI Images from Olympus Slide Scanner as outlined in paper

This program uses a two-step method to detect synaptic boutons on histological images
The first step, a "proposal" algorithm, quickly scans through the image and detects potential boutons by looking for 
groups of pixels that are darker than the surrounding tissue

The second step, a "verification" algorithm, scans each of these potential boutons and verifies them via a pretrained
convolutional neural network.

Command line arguments allow the user to select how the program will search for input images:
-e,--section: Intended for a single histological section.  Required format of this folder is as follows
    Folder
        Image.png  <-- RGB or Greyscale microscopy image.  Should be 345 nm/pixel
        Mask.png <-- Optional Mask with same resolution as Image.png.  Boutons will only be found in white areas
-b,--brain: Intended for an entire brain with many histological sections.  Required format
    Directory
        Section_01
            Image.png
            Mask.png
        Section_02
            Image.png
            Mask.png
        ...
-o,--order: Intended for multiple brains, each with many histological sections.  If the brains are organized as so
    Order.txt
    Brain_01
        Section_01
            Image.jpg
            Mask.jpg
        ...
    Brain_02
        Section_01
            Image.jpg
            Mask.jpg
        ...
    ...
    Then Order.txt should read as so:
    Brain_01
    Brain_02
    ...        

Command line arguments allow the user to select three different types of output.  These output files will be placed in the folder where the Image.png file was located
--png, -p: Program will output a png file with the same dimensions as the image file, with red 5x5 boutons placed on it
--svg, -s: Program will output a svg file with bouton symbols placed on it
--csv, -c: Program will output a csv file with the x,y positions of each bouton
Example useage: python BoutonCount.py -f "Section_1.pn


#Retraining
Our neural network was trained on histological images from our lab, imaged with our microscope. Therefore, retraining the neural network can increase the accuracy of detection. To retrain the neural network, you will first need to calculate the sampling window; the size of the crop fed into the neural network. We have found that a sampling window of approximately 8 microns contains enough context around the bouton to aid in detection without bogging the program down. Because we have a resolution of 345 nm/pixel for our images, this means that our sampling window size was 8 microns / 345 nm/pixel = 24 pixels. Calculate your own sampling window size based on your resolution, and change the value of SIZE in the Constants.py folder

Next, you will require examples of boutons from your tissue, which we will assume is saved as IMAGE. An example IMAGE can be found at Retraining Examples/Example Image.png. Open IMAGE in Photoshop or another similar image editor. Create a second layer, make it entirely white, and title it "Boutons". Set the blending mode to "Darken only", so that the original image is visible again. Set the tool to a red pencil, with a size of 3. Manually click on every punctae that you believe represents a true bouton. When you are finished, save this file as PHOTOSHOP (an example can be found at Retraining Examples/Example.psd). Turn off the visibility of the image layer, so that only the boutons are visible. Save the botuons as LABELS (an example can be found at Retraining Examples/Labels.png).

Next, we will use the "ImageToCSV.py" program to find the locations of boutons. This is a greedy algorithm to identify islands of red dots. Run this as such: "python ImageToCSV.py -f "" -i IMAGE -l LABEL -c CSV", where CSV is the location of the csv file with the locations of boutons (an example can be found at Retraining Examples/Example.csv). If the IMAGE, LABEL, and CSV are in the same folder, the folder command line argument can be used to simplify the execution. For example, if IMAGE = A/Image.png, LABEL=A/Label.png, and CSV=A/Csv.csv, then useage would be "python ImageToCSV.py -f A -i Image.png -l Label.png -c Csv.csv

After this, we will generate negative and positive training data with the "GenerateTrainData.py" program. This will create a numpy array containing all the user selected positive training data, as well as taking blank spots from the image as negative training data. Useage is "python GenerateTrainData.py -c CSV -d "Boutons"".

After this, we will train the neural network with "NetworkTrain.py". This will train a neural network to recognize boutons, and this neural network will be saved to a .json and .h5 file. Useage is "python NetworkTrain.py -t Boutons -m Boutons". Example outputs of this program can be found at Boutons.json and Boutons.h5
