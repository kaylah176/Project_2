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
* matplotlib , seaborn and hvplot for visualization. 
* sklearn.preprocessing for statistical modeling including classification, regression, clustering and dimensionality reduction
* xgboost for energizing machine learning model performance and computational speed
* Pytorch for machine learning
  
## Data Exploration
In the filtering phase, we identified certain categorical features containing irrelevant data that did not align with our parameters. To address this issue, we filtered out data entries that did not specify a certain genre or those designated as global. 

```python
spotify = df.drop(columns = ['Unnamed: 0', 'uri', 'artist_names', 'artist_img', 'artist_individual', 
                             'album_cover', 'artist_id', 'track_name', 'source', 'pivot', 'release_date', 'collab'])
spotify.dropna(inplace = True)
```
During our data exploration phase, we conducted analyses on both numerical and categorical features. The heatmap for numerical features indicated minimal to no correlation between features and streams. For categorical features, we examined `country`, `region`, `artist genre`, and `language`. Subsequently, we narrowed our focus to the top 10 countries and top 10 genres due to the large size of our dataset. 

```python
spotify_filter = spotify_filter.loc[(spotify_filter['country'].isin(top10_country)) &
                                    (spotify_filter['artist_genre'].isin(top10_genre))]
```
<img width="640" alt="Screenshot 2024-04-08 at 6 49 28 PM" src="https://github.com/kaylah176/Project_2/assets/152752672/1f00204f-29a6-4c9d-a660-d99bd5869ef8">


## Section:4  
First off we changed the categorical features to numbers by using dummies for our final data output. When Undersampling the minorty class we used Clustering as a way to identify and seperate groups onto a smaller dataset with two or more variable quantities.

## Model Training  
In this section we used three machine learning algorithms; `Random Forest`, `XGBooster`, and `PyTorch` for undersampling and oversampling. We decided on these alogrithms to train the resampled dataset from section 4 to help with any imbalanced classifications. 

### Random Forest 
* Undersample
```python
rf_under = RandomForestClassifier(random_state = 2, max_features = 'sqrt')
clf_under = GridSearchCV(estimator = rf_under, param_grid = param_grid, cv = 5)
clf_under.fit(X_under_resampled, y_under_resampled)
```

* Oversample
```python
rf_over = RandomForestClassifier(random_state = 2, max_features = 'sqrt')
clf_over = GridSearchCV(estimator = rf_over, param_grid = param_grid, cv = 5)
clf_over.fit(X_over_resampled, y_over_resampled)
```
### XG Booster 
* Undersample
```python
 xgb_clf_under.fit(x_train_xgb, y_train_xgb, eval_set = [(x_valid, y_valid)], verbose = True)
```
* Oversample
```python
  xgb_clf_over.fit(x_train_xgb, y_train_xgb, eval_set = [(x_valid, y_valid)], verbose = True)
```
### PyTorch
* Undersample
```python
X_tensor_under = torch.tensor(X_under_resampled, dtype = torch.float32)  
y_tensor_under = torch.tensor(y_under_resampled, dtype = torch.long)
```
```python
dataset_under = TensorDataset(X_tensor_under, y_tensor_under)  
train_loader_under = DataLoader(dataset_under, batch_size = 64, shuffle = True)
```
* Oversample
```python
X_tensor_over = torch.tensor(X_over_resampled, dtype = torch.float32) 
y_tensor_over = torch.tensor(y_over_resampled, dtype = torch.long)  
```
```python
dataset_over = TensorDataset(X_tensor_over, y_tensor_over) 
train_loader_over = DataLoader(dataset_over, batch_size = 64, shuffle = True)
```
## Section:6 

## Section:7

## Section:8 

## Section:9 

## Conclusion
The last 3 classes has stronger correlation with genre based on factors 

