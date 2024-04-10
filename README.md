# Project_2

## Background
The music industry is a completely new space with the integration of technology. Music previously generated money by selling records and radio airplay. The emergence of online streaming platforms has changed the game for good. It is no longer necessary to buy an entire record to enjoy some songs on an album. Consumers can pick and choose individual songs off any album now. But if that’s the case, how do songs generate money, and what songs seem to be cash cows?

Our project explores the different genres of music and what the revenue implications are for the top ones. We utilize streaming data from Spotify to analyze what the trends are and where the popularity is. We leverage a variety of Python libraries such as `pandas`, `numpy`, `sklearn`, `seaborn`, and `matplotlib` to analyze and visualize the data. We set out to understand numerical and categorical features of the data and separate the relevant information. This is necessary to effectively train, test, and model the data with machine learning. 

## File: 
'Spotify.csv’

## Objective 
* Data Acquisition: Utilize Spotify top 200 songs over the past two years data
* Data Exploration: Leverage libraries like pandas and numpy to manipulate data
* Data Modeling: Train and test split the data to scale, model, and predict it
* Visualization/Performance Analysis: Use matplotlib and hvplot to visualize and better analyze the data

## Hypothesis
We can take the top 200 songs from the last two years on Spotify to train a machine learning model that will produce a forecast of the most streamed genres. This information will be used to predict the amount of revenue certain songs will generate and how much certain artists will earn. This premise places importance on data analytics as one foundation for music production. 

## How to Run the Project 
## 1. Install Packages
* `pandas` for data manipulation;
* `numpy` for numerical operations;
* `matplotlib` , `seaborn` and `hvplot` for visualization. 
* `sklearn.preprocessing` for statistical modeling including classification, regression, clustering and dimensionality reduction
* `xgboost` for energizing machine learning model performance and computational speed
* `Pytorch` for machine learning
  
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

## **4. Model Training**
In this section we used three machine learning algorithms; `Random Forest`, `XGBooster`, and `PyTorch` for undersampling and oversampling. We decided on these alogrithms to train the resampled dataset from section 4 to help with any imbalanced classifications. 

### **4.1 Random Forest** 
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
### XGBooster 
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
Overall, after experimenting with various machine learning algorithms, we concluded that Random Forest and XGBoost performed the best for our model evaluation. However, PyTorch was found to be less suitable for handling our imbalanced classifications.

## 6. Model Evaluation 
Once the model is set and trained, it is time to run the model and evaluate the results. We run both oversampled and undersampled data in the different model approaches (`RandomForest`, `XGBooster`, and `PyTorch`) and analyze the results. The accuracy of the `RandomForest` model was not favorable but not terrible either. The undersample and oversample accuracy were 0.54 and 0.55, respectively. `XGBooster` performed similarly with undersample and oversample accuracy of 0.53 and 0.56, respectively. It was `PyTorch` that really surprised us. Its undersample score was 0.11 while its oversample accuracy was 0.13. A reason for why `RandomForest` and `XGBooster` performed better is due to their ability to better handle categorical and numerical data. 

```python
pred_y_xgb_over = xgb_clf_over.predict(X_test_scaled)
```
<img width="396" alt="Screenshot 2024-04-09 at 7 58 48 PM" src="https://github.com/kaylah176/Project_2/assets/151468004/7f48db54-325e-4c19-8662-12dda745ffc7">


## 7. Feature Importance
From the evaluation, we drew out the most important features of a song. 
The undersample importance is as follows:

![image](https://github.com/kaylah176/Project_2/assets/151468004/560be2d6-8479-4600-b75b-15f709bac083)

The oversample importance is as follows:

![image](https://github.com/kaylah176/Project_2/assets/151468004/b953bd33-b7cc-49c4-8843-086b20277e25)

In both cases, `speechiness`, `acousticness`, `danceability`, and `loudness` are the top four most important features. 

## 8. Deeper Analysis About the Analysis 
We saw what the correlation was in a previous section. In this section we ran correlation analyses on the first seven and last three classes, in order to find out why the predictions accuracy are so different between the first seven and last three classes.

The first seven correlation matrix showed:

![image](https://github.com/kaylah176/Project_2/assets/151468004/5b80138e-3045-4442-9604-c7769f3a3261)

The last three correlation matrix showed:

![image](https://github.com/kaylah176/Project_2/assets/151468004/4be4e59f-93cd-430a-a218-643e25d32c69)

It is immediately visible that there is greater correlation among the last three classes than the first seven. 

## 9. Revenue Forecast 
At this point we have compiled enough data to forecast streams and revenue. 

We use the test dataset to make a comparison between historical average streams and predited average streams per genre.

We concatenated the `X_test` and `y_test` and grouped by genre streams and predicted streams. The historical versus predicted streams are as follows:

<img width="801" alt="Screenshot 2024-04-08 at 9 18 23 PM" src="https://github.com/kaylah176/Project_2/assets/151468004/295c71ff-5451-48bd-8a8f-4cefa91bd64e">

We know a mid-point for revenue per stream is $0.004, so we multiplied this number by the historical and predcted streams to find revenue. That comparison is as follows:

<img width="799" alt="Screenshot 2024-04-08 at 9 21 13 PM" src="https://github.com/kaylah176/Project_2/assets/151468004/36c572bf-98ba-4006-92f5-476f8c8d7a90">


## Conclusion
The last 3 classes has stronger correlation with genre based on factors 
mention https://www.kaggle.com/datasets/yelexa/spotify200
