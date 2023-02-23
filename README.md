# plant_disease_detection
Plant Disease Detection using resMLP Feed-Forward Networks for Image Classification.
ResMLP, also known as the Residual Multi-layer Perceptron, is a neural network architecture that has shown promising results in various computer vision tasks. To use ResMLP for plant disease detection, we can follow the following steps:

**Collect and preprocess the dataset:**  Collect a large dataset of images of healthy and diseased plants, and preprocess them by resizing, cropping, and normalizing the images.

**Split the dataset:** Divide the dataset into training, validation, and testing sets.

**Train the ResMLP model:** Initialize the ResMLP model and train it using the training set. During training, use data augmentation techniques such as random cropping, flipping, and rotation to increase the robustness of the model.

**Tune hyperparameters:** Tune hyperparameters such as learning rate, batch size, and number of training epochs using the validation set to improve the performance of the model.

**Evaluate the model:** Finally, evaluate the performance of the trained model on the testing set by calculating metrics such as accuracy, precision, recall, and F1-score.

Overall, ResMLP can be a powerful tool for plant disease detection, but the quality and size of the dataset, as well as the effectiveness of the data preprocessing and augmentation, can have a significant impact on the performance of the model.


## Collected Various Dataset Leaves

| Blotch Leaves  |  Rot Leaves |
| ------------- | ------------- |
| Necrotic leaf blotch is caused by the rapid synthesis of gibberellins triggered by environmental factors. Treating trees with zinc-containing fungicides (e.g., Ziram) or foliar sprays containing zinc nutrients can decrease the severity of necrotic leaf blotch.  |  Black rot is caused by the fungus Diplodia seriata (syn Botryosphaeria obtusa). The fungus can infect dead tissue as well as living trunks, branches, leaves and fruits. The black rot fungi survive Minnesota winters in branch cankers and mummified fruit (shriveled and dried fruit) attached to the tree.  |



| Scab Leaves  | Healthy Leaves |
| ------------- | ------------- |
| Leaves infected with the apple scab fungus usually fall from trees in autumn or early winter. The fungus continues to live within the leaves during winter, forming small, flask-shaped bodies, in which spores (ascospores) develop. These ascospores mature in spring and are forcibly ejected during spring rains  | The leaves are simple, oval in shape, have small serrations along the margin, and are arranged alternately along the branches. A typical leaf is 2 – 5 inches long, 1.5 – 2.5 inches wide, and has an acuminate tip (Petrides 1972). The twigs, buds, and undersides of the leaves usually have white pubescence  |
