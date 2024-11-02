
## Task 1 - Linear and Logistic Regression - HelloWorld for AIML
### Linear Regression
Linear regression is a supervised machine-learning algorithm that learns from the labelled datasets and maps the data points to the most optimized linear functions, which can be used for prediction on new datasets. First off we should know what supervised machine learning algorithms is. It is a type of machine learning where the algorithm learns from labelled data.  Labeled data means the dataset whose respective target value(y) is already known.   
#


The goal of the linear regression algorithm is to find the best Fit Line equation that can predict the values based on the independent variables.  

#  

In regression set of records are present with X and Y values and these values are used to learn a function so if you want to predict Y from an unknown X this learned function can be used. In regression we have to find the value of Y, So, a function is required that predicts continuous Y in the case of regression given X as independent features.    

# 

Later in task 5,I learned it from scratch.For a basic idea I  predicted the price of perth houses based on multiple different variables using sci-kit’s linear_model.LinearRegression().
#

Here's the link for the kaggle notebook - [Linear Regression](https://www.kaggle.com/code/ashith1709/linear-regression)  

### Logistic Regression


**Logistic Regression** is a statistical model commonly used for binary classification tasks, where the goal is to categorize data points into one of two classes. (e.g. predicting presence/absence of disease)

#### Key Concepts

1.**Sigmoid Function**:  
   Logistic regression uses the sigmoid (or logistic) function to map predictions to probabilities. The sigmoid function is defined as:
   
σ(z) = 1 / (1 + e^(-z))

where z = w * x + b.
#


2.**Probability Output**:  
   The output of the sigmoid function is a probability between 0 and 1. Logistic regression classifies based on a threshold (often 0.5):
   - If \( P(y=1|x) > 0.5 \), the prediction is class 1.
   - If \( P(y=1|x) <= 0.5 \), the prediction is class 0.
#


3.**Cost Function**:  
   Logistic regression uses the **binary cross-entropy** loss, which penalizes incorrect predictions. This cost function is minimized during training to improve model performance.

Cost = -(1/N) * Σ [yᵢ * log(h(xᵢ)) + (1 - yᵢ) * log(1 - h(xᵢ))]

   where \( h(x_i) \) is the predicted probability for each example.

#


4.**Training Process**:  
   **Gradient Descent** is used to find optimal parameters \( w \) and \( b \) that minimize the cost function.The algorithm iteratively updates these parameters, improving model accuracy.   
#
Later in task 5,I learned it from scratch.For a basic idea I trained a model to distinguish between different species of the Iris flower based on sepal length, sepal width, petal length, and petal width using sci-kit’s linear_model.LogisticRegression.  
#
Here's the link for the kaggle notebook - [Logistic Regression](https://www.kaggle.com/code/ashith1709/logistic-regression)  

## Task 2 - Matplotlib and Data Visualisation  
  
**Matplotlib** is a popular Python library used for data visualization. It provides tools to create a wide variety of static, animated, and interactive plots, ranging from simple line plots to complex multi-dimensional graphs.Data visualization is crucial for interpreting data trends, spotting anomalies, and conveying insights effectively.
# 

 Matplotlib allows detailed control over plot elements like colors, labels, styles, and grid layouts.It Works well with other Python libraries, such as **Pandas** for data manipulation and **NumPy** for numerical operations.It offers line plots, scatter plots, bar plots, histograms, box plots, pie charts, heatmaps, violin plots, area plots, 3D plots,multiple sub plots, stem plots, quiver plots, and step plots.
#


Matplotlib also has a user-friendly interface through **Pyplot**, which provides MATLAB-like commands that simplify plot creation, making it accessible even for beginners. 
#

The task was to explore the various basic characteristics to plots with python libraries,make a multivariate distribution for the given dataset for a classification task and to understand an elementary idea of clustering.

![plot](https://raw.githubusercontent.com/ashith-17/Marvel-Level-01/refs/heads/main/matplo.png)
#
Here's the link for the kaggle notebook-[Matplotlib and Data Visualisation](https://www.kaggle.com/code/ashith1709/matplotlib-and-data-visualisation)

#

## Task 3 - Numpy

**NumPy** (Numerical Python) is a powerful Python library used for numerical and scientific computing. It provides support for large, multi-dimensional arrays and matrices, along with a wide range of mathematical functions like addition, subtraction, trigonometry, statistics, and linear algebra on these arrays. 
#
The core feature of NumPy is its array object (`ndarray`), which allows for fast array operations and manipulation.It provides flexible methods for accessing and modifying subsets of arrays and facilitates operations on arrays of different shapes by expanding dimensions to match.It Works seamlessly with libraries like **Pandas**, **Matplotlib**, and **SciPy**.

#

The task was to generate an array by repeating a small array across each dimension and to generate an array with element indexes such that the array elements appear in ascending order.

![Numpy](https://raw.githubusercontent.com/ashith-17/Marvel-Level-01/refs/heads/main/Numpy.png)

#

Here's the link for the code-[Numpy](https://www.kaggle.com/code/ashith1709/numpy)



## Task 4 - Metrics and Performance Evaluation

In both regression and classification, metrics are used to evaluate the model's performance by assessing how well the predicted values align with the actual data.


### Regression Metrics

The metrics focus on assessing the accuracy of the predictions relative to the actual values.
#

1.**Mean Absolute Error (MAE):**
  The average absolute difference between predicted and actual values.Gives a sense of the average prediction error. Lower values indicate better performance.
#


2.**Mean Squared Error (MSE):**
  The average of the squared differences between predicted and actual values. Sensitive to larger errors due to squaring, which penalizes significant deviations more heavily.
#


3.**Root Mean Squared Error (RMSE):**
  The square root of the MSE, providing an error metric in the same units as the target variable.Similar to MSE but more interpretable since it's in the same unit as the data.
#


4.**R-squared (R²):**
   Represents the proportion of the variance in the target variable explained by the model.A value between 0 and 1, where 1 indicates that the model explains all the variance in the data.
#


5.**Adjusted R-squared:**
   A modified R² that accounts for the number of predictors in the model.Useful when comparing models with different numbers of predictors, as it penalizes models with unnecessary predictors.
#
Here's the code implementing the Regression metrics-[Regression metrics](https://www.kaggle.com/code/ashith1709/regression-metrices)


### Classification Metrics

The metrics evaluate the model’s ability to classify correctly.
#


1.**Accuracy:**
   The proportion of correct predictions (both true positives and true negatives) out of the total predictions.Works well when classes are balanced but can be misleading in imbalanced data.
#


2.**Precision:**
    The proportion of true positive predictions among all positive predictions.Indicates the accuracy of positive predictions, useful when false positives are costly.
#


3.**Recall (Sensitivity or True Positive Rate):**
    The proportion of true positives among all actual positives.Shows how well the model captures actual positives, valuable when false negatives are critical.
#


4.**F1 Score:**
    The harmonic mean of precision and recall, providing a balance between the two.Useful for imbalanced classes as it penalizes extreme precision or recall values.
#


5.**ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
   A measure of the model's ability to distinguish between classes by plotting the True Positive Rate vs. False Positive Rate at various thresholds.An AUC of 1 indicates perfect separation between classes, while 0.5 indicates no discriminative power.
#


6.**Log Loss (Binary Cross-Entropy):**
    Measures the performance of a classification model by penalizing wrong predictions based on their confidence.Lower values indicate better model performance.
#
Here's the link implementing the Classification metrics-[Classification Metrics](https://www.kaggle.com/code/ashith1709/classification-metrices)











