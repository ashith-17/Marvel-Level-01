
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
  
**Matplotlib** is a popular Python library used for data visualization. It provides tools to create a wide variety of static, animated, and interactive plots, ranging from simple line plots to complex multi-dimensional graphs. Matplotlib is particularly valued for its flexibility and control over plot appearance, making it a preferred choice for detailed customization in professional and academic visualizations.Data visualization is crucial for interpreting data trends, spotting anomalies, and conveying insights effectively.
# 

 Matplotlib allows detailed control over plot elements like colors, labels, styles, and grid layouts.It Works well with other Python libraries, such as **Pandas** for data manipulation and **NumPy** for numerical operations.It offers line plots, scatter plots, bar plots, histograms, box plots, pie charts, heatmaps, violin plots, area plots, 3D plots,multiple sub plots, stem plots, quiver plots, and step plots.
#


Matplotlib also has a user-friendly interface through **Pyplot**, which provides MATLAB-like commands that simplify plot creation, making it accessible even for beginners. This capability has made Matplotlib a fundamental tool for data analysis and visualization in Python.
#

The task was to explore the various basic characteristics to plots with python libraries,make a multivariate distribution for the given dataset for a classification task and to understand an elementary idea of clustering.
#
Here's the link for the kaggle notebook-[Matplotlib and Data Visualisation](https://www.kaggle.com/code/ashith1709/matplotlib-and-data-visualisation)







