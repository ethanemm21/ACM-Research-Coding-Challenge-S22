# ACM Research Coding Challenge (Spring 2022)

## Libraries
To begin the challenge, I had to fully understand what and how binary classification worked. I read the binary classification Wikipedia, and I decided to use Machine Learning, specifically TensorFlow Keras, to solve the problem of classifying mushrooms as edible or poisonous. Additionally, to help sort the data, I used pandas and NumPy. Finally, I used scikit-learn, to split data into training and testing splits. 

## Approach
Although I was not well-versed in Keras or scikit-learn, I remember a [YouTube video](https://www.youtube.com/watch?v=z1PGJ9quPV8&t=583s&ab_channel=Khanrad) I watched, which classified a cancer data set into malign or benign tumors based off various characteristics of the tumor. I recognized the similarity between these problems, and I used a similar algorithm to solve the binary classification of `mushrooms.csv`.

### Machine Learning
Using scikit-learn, I used the train_test_split function to split the original data into training and testing splits. By doing this, I would be able to evaluate the validity of the model, not only on the data it's been trained on, but new, test data. I chose to split the data with 80% to train, and 20% to test. 

I created a Sequential model from Keras, which is a model where you can add neural network layers to the machine learning algorithm. From this, I added a Dense, or dense layer, where I was able to choose the amount of neurons, input shape, and activation function. For my activation function, I chose to use the "sigmoid" function, which is a function that returns a value between 0 and 1 for all values in the neural network. This is especially useful for binary classification, as it simplifies the model. Finally, for the last layer, I included 1 Dense neuron, which is the final returned value of 0 or 1 for "edible" or "poisonous" mushrooms.  

Next, I compiled the data. For the optimizer, I chose "adam," which implements the ["Adam algorithm"](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/). Although I didn't fully understand how the algorithm worked, I understood that it was efficient for large data sets, returned results quickly, and was commonly used for machine learning. For the loss parameter, I chose binary crossentropy, as that is the ["default loss function"](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/) for binary classification. I then chose accuracy as the metric displayed, to see how accurate the model was. 

Finally, to fit the data from `mushrooms.csv`, I fit the data with the training set, and I added a validation split of 20%. By utilizing a validation split, this helps the model not to overfit to the training set, as it validates the model that has been trained. Next, I chose the epochs, or iterations, to be 50, but it can be increased or decreased for the model to become more or less accurate. By around the 21st epoch, the the model consistently has an accuracy of 1.000 on both the training and validation sets. I then evaluted the model with the test splits. With the test splits, the model had an accuracy of 1.000 with a minimal loss. 

### Sorting Data
I realized that in `mushrooms.csv`, all the data was in the form of letters. However, when I first approached the problem, I didn't realize the Keras ML algorithm wouldn't work. Therefore, I had to change my data in order to to be numerical data, which the model could train, validate, and test. I added a function which modified the data to fit the model, which I got from [this article](https://pythonprogramming.net/working-with-non-numerical-data-machine-learning-tutorial/). I was then able to successfully test the model. 

## Other Sources used to learn (if not previously mentioned)
[TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)  
[Keras API Reference](https://keras.io/api/)  
[YouTube Explanation](https://www.youtube.com/watch?v=Zi-0rlM4RDs&ab_channel=deeplizard) - Helped me understand the difference between training, validation, and testing splits  
[Binary Classification Tutorial 1](https://www.atmosera.com/blog/binary-classification-with-neural-networks/)  
[Binary Classification Tutorial 2](https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/)



