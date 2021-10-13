###########################################################
################# Machine learning using R ################
###### Copyright - HURU School Data Science Dpt. 2021 #####
###########################################################
# Caret is the short for Classification And REgression Training. It is a complete package that covers all the stages of a pipeline for creating a machine learning predictive model. We will cover the folowing:
# How to install caret

install.packages("caret")

data(mtcars)    # Load the dataset
head(mtcars)
?mtcars         # Get more information about this dataset
# How to create a simple model
library(caret)
library(ggplot2)
library(lattice)

# The train() function has three basic parameters:
# Formula
# Dataset
# Method (or algorithm): https://topepo.github.io/caret/available-models.html


# Simple linear regression model (lm means linear model)
model <- train(mpg ~ wt,
               data = mtcars,
               method = "lm")
model
# Multiple linear regression model
model <- train(mpg ~ .,
               data = mtcars,
               method = "lm")

model
# Ridge regression model
model <- train(mpg ~ .,
               data = mtcars,
               method = "lasso") # Try using "lasso"

model
# How to use cross-validation to avoid overfitting
# Resampling procedure to evaluate machine learning models with limited data
## 10-fold CV
# possible values: boot", "boot632", "cv", "repeatedcv", "LOOCV", "LGOCV"

fitControl <- trainControl(method = "repeatedcv",   
                           number = 10,     # number of folds
                           repeats = 10)    # repeated ten times

model.cv <- train(mpg ~ .,
                  data = mtcars,
                  method = "lasso",  # now we're using the lasso method
                  trControl = fitControl)  

model.cv 
#shows you the 
# R squared  - 
# MAE(Mean Absolute Error) and RMSE (Root Mean Square Error)-  lower values are better - we want to get closer to 0
# more on these metrics - https://www.youtube.com/watch?v=lHAEPyWNgyY
# which all show the prediction errors and model accuracy


# How to add Feature Engineering / simple preprocessing to your data for parameter tuning
# Center data (i.e. compute the mean for each column and subtracts it from each respective value);
# Scale data (i.e. put all data on the same scale, e.g. a scale from 0 up to 1)
# Other options: “BoxCox”, “YeoJohnson”, “expoTrans”, “range”, “knnImpute”, “bagImpute”, “medianImpute”, “pca”, “ica” and “spatialSign”

model.cv <- train(mpg ~ .,
                  data = mtcars,
                  method = "lasso",
                  trControl = fitControl,
                  preProcess = c('scale', 'center')) # default: no pre-processing

model.cv
# How to find the best parameters for your chosen model

# Here I generate a dataframe with a column named lambda with 100 values that goes from 10^10 to 10^-2

lambdaGrid <- expand.grid(lambda = 10^seq(10, -2, length=100))

model.cv <- train(mpg ~ .,
                  data = mtcars,
                  method = "ridge",
                  trControl = fitControl,
                  preProcess = c('scale', 'center'),
                  tuneGrid = lambdaGrid,   # Test all the lambda values in the lambdaGrid dataframe
                  na.action = na.omit)   # Ignore NA values

model.cv # Will take sometime to play with all the parameters and select the best model

# How to see the most important features/variables for your model
ggplot(varImp(model.cv))

# Why engine displacement matters
# Engine displacement is a determining factor in the horsepower and torque that an engine produces, as well as how much fuel that engine consumes. Generally speaking, the higher an engine’s displacement the more power it can create, while the lower the displacement the less fuel it can consume. That makes sense as mpg is affected greatly by displacement.

# How to use your model to predict

# import data from a Subaru Outback and Landcruiser VX
test <- read.csv(file='mpg_test.csv')

head(test)
# Use the caret predict function to make any predictions
predictions <- predict(model.cv, test)

# Here we can now make good predicitons on the miles per gallon

predictions

#################################################################
### k Means clustering #########################################
#stats::kmeans(x, centers = 3, nstart = 10)
i <- grep("Length", names(iris))
x <- iris[, i]
cl <- kmeans(x, 6, nstart = 2)
plot(x, col = cl$cluster)

#NOTE: How to determine the number of clusters
##Run k-means with k=1, k=2, …, k=n
##Record total within SS for each value of k.
##Choose k at the elbow position

###################################################################
### Decision Trees #######################
#A great advantage of decision trees is that they make a complex decision simpler by breaking it down into smaller,
#simpler decisions using a divide-and-conquer strategy. 
#They basically identify a set of if-else conditions that split the data according to the value of the features.

install.packages(c("rpart","rpart.plot", "mlbench", "ranger"))

library("mlbench")#Used to read in the Sonar dataset

data(Sonar)
View(Sonar)

library("rpart") ## recursive partitioning
m <- rpart(Class ~ ., data = Sonar,
           method = "class")
library("rpart.plot")
rpart.plot(m)

p <- predict(m, Sonar, type = "class")
table(p, Sonar$Class)

#Training a random forest
set.seed(12)
model <- train(Class ~ .,
               data = Sonar,
               method = "ranger")
print(model)

plot(model)
##############################################################################
##### Logistic Regression ###################################################
# Load the data and remove NAs
library(dplyr)

data("PimaIndiansDiabetes2", package = "mlbench")
PimaIndiansDiabetes2 <- na.omit(PimaIndiansDiabetes2)

# Inspect the data
head(PimaIndiansDiabetes2, 3)

# Split the data into training and test set
set.seed(123)

training.samples <- PimaIndiansDiabetes2$diabetes %>% 
  createDataPartition(p = 0.8, list = FALSE)

train.data  <- PimaIndiansDiabetes2[training.samples, ]

test.data <- PimaIndiansDiabetes2[-training.samples, ]

#Quick logistic regression
# Fit the model
model <- glm(diabetes ~., data = train.data, family = binomial)
# Summarize the model
summary(model)
# Make predictions
probabilities <- model %>% predict(test.data, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy
mean(predicted.classes == test.data$diabetes)

##Fitting the modelusing the glusoce column
model <- glm(diabetes ~ glucose, data = train.data, family = binomial)
summary(model)$coef

#Using the model
newdata <- data.frame(glucose = c(20,  180))
probabilities <- model %>% predict(newdata, type = "response")
predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
predicted.classes

#Plotting the predictions
train.data %>%
  mutate(prob = ifelse(diabetes == "pos", 1, 0)) %>%
  ggplot(aes(glucose, prob)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(
    title = "Logistic Regression Model", 
    x = "Plasma Glucose Concentration",
    y = "Probability of being diabete-pos"
  )

###########################################################
################# BIG DATA ML ################
###### Using Machine Learning on distributed data systems #####
###########################################################

# Import necessary libraries
library(sparklyr)
library(ggplot2)
library(dplyr)

# Connect to spark cluster - in our case we will builtd local cluster and copy some data into the clusters

sc <- spark_connect(master = "local")

mtcars_tbl <- copy_to(sc, mtcars, "mtcars", overwrite = TRUE)

# transform our data set, and then partition into 'training', 'test'
partitions <- mtcars_tbl %>%
  filter(hp >= 100) %>%
  sdf_random_split(training = 0.6, test = 0.4, seed = 888)

# fit a linear model to the training dataset
fit <- partitions$training %>%
  ml_linear_regression(mpg ~ wt + cyl)

# can use other algorithms based on need:
# https://spark.rstudio.com/mlib/

# summarize the model
summary(fit)


# Score the data
pred <- ml_predict(fit, partitions$test) %>%
  collect

# Plot the predicted versus actual mpg
ggplot(pred, aes(x = mpg, y = prediction)) +
  geom_abline(lty = "dashed", col = "red") +
  geom_point() +
  theme(plot.title = element_text(hjust = 0.5)) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Actual Fuel Consumption",
    y = "Predicted Fuel Consumption",
    title = "Predicted vs. Actual Fuel Consumption"
  )
