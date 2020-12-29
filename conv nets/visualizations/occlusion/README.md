### Description
These are some examples resulting from the occlusion experiments.

The left-most image shows a heatmap where brigther regions represent portions of the input for which the classifier output is really "sensible to", i.e., the different between the output when the original input is given and the output when the modified (occluded) image is given is big!.

The middle image is the orignal image.

The righ-most image is the original image occluded according to the heatmap. In other words, the "most important" regions are visible and viceversa.

Notice that every image with a different occluded region is a new input that most be evaluated with the model. To improve the throughput time of the experiment a [tf.data pipeline](/conv%20nets/visualizations/occlusion/occlusion.py#L90) is created to generate batches of images with occluded regions. A general overview of the process is:
  1. Sample a image from the original dataset.
  2. Repeat this image as many times as occluded images you will have.
  3. Enumerate each occluded image.
  4. Create and apply an occluded mask based on this enumeration (changing location of course!). An example of how the masks are generated can be found [here](/conv%20nets/visualizations/occlusion/occlusion_mask_generation_test.ipynb).
  5. Create batches.
  6. Run model on batches and compute sensitivity.

### Results
![](/conv%20nets/visualizations/occlusion/example1.png)
![](/conv%20nets/visualizations/occlusion/example2.png)
![](/conv%20nets/visualizations/occlusion/example3.png)
![](/conv%20nets/visualizations/occlusion/example4.png)
