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
* pandas for data manipulation;
* numpy for numerical operations;
* yfinance for fetching assets prices;
* matplotlib , seaborn and hvplot for visualization. 
* sklearn.preprocessing for statistical modeling including classification, regression, clustering and dimensionality reduction
* xgboost for energizing machine learning model performance and computational speed
* Pytorch for machine learning
  
## 2. Set Up Global Parameters
In this initial phase, we established the foundation for our analysis by defining global parameters. These parameters include the date range for our data, the list of genres to analyze alongside its ranking, and any financial metrics of interest (e.g. number of streams, price of each stream). These steps are crucial as it ensures that all subsequent analyses operate under a consistent set of assumptions and data scope.
Through our filtering we found that some of our categorical features had some unnecessary data that did not fit under our parameters. To fix this we filtered the data that did not have a name for the type of genre or data that was taken globally (not within our top 10 categories).  

spotify_filter = spotify.loc[(spotify['artist_genre'] != '0') & (spotify['country'] != 'Global') & (spotify['language'] != 'Global')]

## 3. Data Exploration
In our data exploration phase, we concentrated on both numerical and categorical features to investigate potential correlations with streams. However, through our analysis the heatmap revealed minimal to no correlation between features and streams. Following this realization, we opted to narrow down our data scope to four key features: country, region, genre, and language. This refinement will help our dataset, making it more manageable for further analysis in the following section.

## Section:4  
First off we changed the categorical features to numbers by using dummies for our final data output. When Undersampling the minorty class we used Clustering as a way to identify and seperate groups onto a smaller dataset with two or more variable quantities.

## Section:5 
In this section we used Random Forest in order to find an algorithm within our new data set in order to find a connection between the number of steams and features. Hence, why we chose this machine learning because it give us accurate and precise results. However to make our claim stronger we added XGBooster to our code for a gradient boosting algorithm that can be used for classification and making predictions as well. 

Ask more about the code in this section. -Kim

## Section:6 

## Section:7

## Section:8 

## Section:9 

## Conclusion
The last 3 classes has stronger correlation with genre based on factors 

