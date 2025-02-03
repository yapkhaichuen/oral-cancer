In this application, we propose and implement a convolutional neural network (CNN) framework for the binary classification of real-world images into “cancer” and “non-cancer” categories. The methodology encompasses a robust pipeline that includes data augmentation, stratified dataset splitting, model architecture design, and performance evaluation. Our experiments, conducted on a curated dataset with 87 cancer and 44 non-cancer images, demonstrate a moderate classification accuracy of approximately 66.67% on unseen test data. Despite the small dataset size, our work underscores the potential of deep learning methods in limited-conditions medical image analysis and lays the groundwork for further improvements via model tuning, increased data availability, and advanced augmentation strategies.


**2. Materials and Methods**

**2.1 Data Preparation and Augmentation**  
The dataset, stored under the directory `./image_data`, comprises two classes: `cancer` (87 images) and `non-cancer` (44 images). The function `GetDatasetSize` iterates through the dataset directories to determine the number of images per class. Subsequently, the data is partitioned into training (70%), validation (15%), and testing (15%) subsets using the `TrainValTestSplit` function. This stratified split ensures that the model is trained on a representative sample while reserving adequate data for validation and testing.

To enhance model generalizability, data augmentation is applied via Keras’ `ImageDataGenerator`. The training data is enriched with random zoom, shear, and horizontal flip transformations along with normalization (rescaling pixel values to [0,1]). Separate generators without augmentation are used for validation and testing to maintain data integrity during evaluation.

**2.2 Model Architecture**  
The CNN model is constructed using Keras’ Sequential API. The architecture is comprised of multiple convolutional layers with increasing filter depths (32, 64, 128) interspersed with max-pooling layers to reduce spatial dimensionality. Dropout layers are incorporated to mitigate overfitting, and a final fully connected (dense) layer with a sigmoid activation function outputs the probability of an image belonging to the cancer class. The model is compiled with the Adam optimizer and binary crossentropy loss, which is suitable for binary classification tasks.

The summary of the network is as follows:
- **Input Layer:** 256×256×3 images.
- **Convolutional Layers:** Multiple layers with ReLU activations, gradually increasing filter counts.
- **Pooling and Dropout:** Interleaved to reduce overfitting and computational complexity.
- **Flattening and Dense Layers:** Convert feature maps into a vector and process through dense layers.
- **Output Layer:** A single neuron with sigmoid activation for binary prediction.

**2.3 Training and Checkpointing**  
The model is trained over 32 epochs with periodic validation using a minimal number of steps per epoch. A `ModelCheckpoint` callback is implemented to save the best-performing model based on validation accuracy. The training history, including loss and accuracy metrics for both training and validation sets, is saved for subsequent analysis and plotting.

**2.4 Evaluation and Prediction**  
After training, the best model is loaded from the saved checkpoint file. The model is then evaluated on the testing subset, yielding an accuracy of approximately 66.67%. Furthermore, the script includes an inference function `cancerPrediction` that accepts a file path, processes the input image, and outputs the predicted class (Cancer or Non-Cancer) by thresholding the sigmoid output at 0.5.

---

**3. Results and Discussion**  
The CNN model, when trained on the available dataset, achieved a testing accuracy of 66.67%. While this level of performance indicates the model’s capacity to learn discriminative features, the moderate accuracy also highlights the challenges associated with small datasets and the need for further hyperparameter optimization. The training and validation accuracy/loss curves (as plotted using Matplotlib) provide insights into the learning dynamics, suggesting potential avenues for further investigation such as deeper architectures, additional regularization techniques, and the inclusion of more diverse training data.

---

**Acknowledgments:**  
We acknowledge the contributions of open-source libraries such as OpenCV, TensorFlow, and Keras, which have been instrumental in developing and validating the proposed framework.