# 19Fall_MSBD5001_IndividualPro
The codes and the description of my own project for 19Fall MSBD5001 Individual Project in HKUST BDT Program


There are two different model to solve with this problem.
The difference is 'whether using a classificator to predict if the user will play one game in testing dataset'.

It is same to process the rough dataset into training set and testing set, I combine all of 'attribute' words within variables 'genres', 'categories' and 'tags' and transfer them as dummy variables.

Moreover,
Processing the NA: delete 4 Na sample.
Transfer the time varibles 'release date' and 'purchase date' via calculating the difference of this two time-dtype variable, which means create a new varibles 'date_diff': 'purchase date'-'release date'.

And for Method 2, I just use 5 variables to predict the play_time, without those dummy variable, and the model is simple linear model

And for Method 1, firstly I combine those 5 variables with dummy variables, and then using GBDT as a classificator to predict whether the user will play one game, which means I consider there is a new label 'play_or_not' for samples in training set, if the 'playtime_forever'==0, this variable will be 0, otherwise, it will be 1. 

Then training a simple linear regression model via the subset of training set with 'play_or_not'==1, and predicting the 'playtime_forever' for testing data which is predicted will be played by user
