# Project_2
Objective: Determining contributing factors to Spotify revenue streams

## Background
The music industry is a completely new space with the integration of technology. Music previously generated money by selling records and radio airplay. The emergence of online streaming platforms has changed the game for good. It is no longer necessary to buy an entire record to enjoy some songs on an album. Consumers can pick and choose individual songs off any album now. But if that’s the case, how do songs generate money, and what songs seem to be cash cows?

Our project explores the different genres of music and what the revenue implications are for the top ones. We utilize streaming data from Spotify to analyze what the trends are and where the popularity is. We leverage a variety of Python libraries such as pandas, numpy, sklearn, seaborn, and matplotlib to analyze and visualize the data. We set out to understand numerical and categorical features of the data and separate the relevant information. This is necessary to effectively train, test, and model the data with machine learning. 

## File: 
'Spotify.csv’

## Objective 
Data Acquisition: Utilize Spotify top 200 songs over the past two years data
Data Exploration: Leverage libraries like pandas and numpy to manipulate data
Data Modeling: Train and test split the data to scale, model, and predict it
Visualization/Performance Analysis: Use matplotlib and hvplot to visualize and better analyze the data

## Hypothesis
We can take the top 200 songs from the last two years on Spotify to train a machine learning model that will produce a forecast of the most streamed genres. This information will be used to predict the amount of revenue certain songs will generate and how much certain artists will earn. This premise places importance on data analytics as one foundation for music production. 


## How to Run the Project 
## 1. Install Packages
pandas for data manipulation;
numpy for numerical operations;
yfinance for fetching assets prices;
matplotlib , seaborn and hvplot for visualization. 
sklearn.preprocessing for statistical modeling including classification, regression, clustering and dimensionality reduction
xgboost for energizing machine learning model performance and computational speed
Pytorch for machine learning 
## 2. Set Up Global Parameters
In this initial phase, we establish the foundation for our analysis by defining global parameters. These parameters include the date range for our data, the list of genres to analyze alongside its ranking, and any financial metrics of interest (e.g. number of streams, price of each stream). This step is crucial as it ensures that all subsequent analyses operate under a consistent set of assumptions and data scope.
Through our filtering we found that some of our categorical features had some unnecessary data that did not fit under our parameters. To fix this we filtered data that it not have a name for the type of genre the music was or data that was taking data globally (not within our top 10 categories).  

spotify_filter = spotify.loc[(spotify['artist_genre'] != '0') & (spotify['country'] != 'Global') & (spotify['language'] != 'Global')]

## 3. Data Exploration
Feature Preprocessing 
This section is a thorough examination of the dataset, employing statistical analyses, and visualizations to uncover the relationship with genre and streams.
## 4.1 
First off changing the categorical features to numbers by using dummies for our final data output. Doing this will allow us to change categorical features using dummies 
Final data all numbers 
## Conclusion
The last 3 classes has stronger correlation with genre based on factors 

