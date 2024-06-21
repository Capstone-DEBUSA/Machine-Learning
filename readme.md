# Machine-Learning

# Function Dependencies

| Library     | Version    | 
|------------ |------------|
|Tensorflow	  |<code>^2.5.0</code>|
|Keras	      |<code>^2.4.3</code>|
|Matplotlib	  |<code>^3.4.2</code>|
|NumPy	      |<code>^1.19.5</code>|
|Pandas	      |<code>^1.2.4</code>|
|Scikit-learn	|<code>^0.24.2</code>|
|Seaborn      |<code>^0.11.1</code>|

# Transfer Learning ResNet152V2
ResNet152V2 is a deep learning model used for image classification, including the classification of ten types of vegetables and fruits. It is pre-trained on a large dataset and can extract relevant features from vegetables and fruits images. The model's architecture includes convolutional layers, pooling layers, and fully connected layers. It is trained using labeled fish images, and can classify new images by assigning probabilities to each fish species. ResNet152V2 is effective for accurately identifying vegetables and fruits based on visual characteristics.

# Dataset
We obtained the dataset by independently photographing vegetables and fruits, making it our original data. The dataset was then split into a training set and a test set. The training set was used to train the ResNet152V2 model, while the test set was used to evaluate its performance.

| Dataset     |
|------------ |
| Apel        |
| Brokoli     |
| Jeruk       |
| Kangkung    |
| Mangga      |
| Pisang      |
| Strawberry  |
| Terong      |
| Toge        |
| Wortel      |

![bar chart of train each category](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/7550fdb7-6dca-45b1-afe4-300016135747)
![bar chart of uji image each category](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/6265907d-6e22-4457-a1f7-01a0dcb8e811)

# Model Architecture
ResNet152V2 is a state-of-the-art deep learning model designed specifically for image classification tasks. It encompasses a sophisticated architecture comprising residual blocks, convolutional layers, and pooling layers. This architecture enables the extraction of hierarchical features at varying levels of abstraction, facilitating the accurate classification of various objects and patterns.
In addition to its primary architecture, ResNet152V2 incorporates pre-activation residual blocks, where batch normalization and activation functions (ReLU) are applied before the convolutional operations. These pre-activation residual blocks, along with shortcut connections, allow gradients to propagate more effectively through the network. These features mitigate the vanishing gradient problem during training, enhancing the overall performance and convergence speed of the model.

# Training
Train the ResNet152V2 model using the labeled images within the training set. Utilize prominent machine learning frameworks or libraries, such as TensorFlow or Keras, which provide pre-built implementations of ResNet152V2. Fine-tune hyperparameters and training configurations based on experimentation and model performance.

# Pre-built implementations of ResNet152V2
Download the pre-trained weights. No top means it excludes the fully connected layer it uses for classification.
!wget --no-check-certificate \
https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5

# Fine-tune hyperparameters and Training configurations
| Type    | Value    |
|------------|------------|
|Learning Rate                | <code>0.0001</code>| 
|Optimizer                    | <code>Adam</code>| 
|Batch Size                   | <code>32</code>| 
|Number of Training Epochs    | <code>10</code>| 
|Input Shape                  | <code>(416,416,3)</code>| 
|Data Augmentation Parameters | <code>rescale=1./255</code>| 
|Regularization Techniques    | <code>layers.GlobalMaxPooling2D()(ResNet152V2_last_output)</code><br><code>layers.Dense(512, activation='relu')(x_ResNet152V2)(x)</code><br><code>layers.Dropout(0.15)(x_ResNet152V2)(x)</code><br><code>layers.Dense(10, activation='softmax')(x_ResNet152V2)</code><br>| 

# Evaluation and Visualitation
Once the model training is complete, evaluate its performance using the test set. Measure accuracy and other relevant evaluation metrics to assess the model's classification capability.

# Model Accuracy & Lose
loss: 8.6371e-04 - accuracy: 1.0000 - val_loss: 0.0015 - val_accuracy: 1.0000
![accuration](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/0acfe51c-2bf5-4751-8ec8-6567002a4ef1)
![loss](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/f4849ae3-91db-4f20-95c0-285f02487f5d)

# Classification Report at Test Dataset
![confusion matrix val data](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/7f11ac09-4549-431c-85d5-430bc8dab0a4)

# Confusion Matrix at Test Dataset
![predict model to validation data](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/f78bd0e9-b9e9-42ae-aaf3-d9794ab348ee)

# Example Prediction
![display labels and prediction validation data](https://github.com/Capstone-DEBUSA/Machine-Learning/assets/99036085/81e4e781-4efd-4f45-bce9-fce0bdd71466)
