# Air-BnB-Price-Prediction

## Introduction
Airbnb is an online platform that allows homeowners and renters to put their properties online primarily as vacation rentals and homestays and has seen tremendous growth since its inception in August 2013 in San Francisco California. In December of 2020 Airbnb became a public company with an IPO that raised $3.5 billion, with stocks started the day at $68 a share and ending at $144.71 [1]. With competitors like VRBO and others also joining the niche home sharing market as well as hotel and other options there is a lot of competition to secure quests and keep them.

On Airbnb hosts set their own prices for their listings, although Airbnb and similar home-sharing sites provide recommendations based on a variety of factors, they cannot give an optimal price recommendation for each property that is listed on the platform. There are third-party options for hosts, however these require the hosts to input a base price to give any advice on seasonal or promotional pricing and cut into their bottom line. Airbnb pricing is an important aspect to each host as it can drastically change the number of guests each property has, especially in a crowded market.

Given the considerable number of options that customers have, and that 53% of respondents to the Morgan Stanley study use Airbnb because of the cost savings, this means that over half of Airbnb rentals are happening because of their price [2]. This coupled with the fact that only 11% of listings are reserved by customers on a typical night means that hosts who are not optimizing their pricing strategy are losing out on a lot of money, and there is clearly room for improvement [2]. This paper will attempt to solve this problem by using machine learning techniques with Python to predict the price bin for properties in six major cities in the United States.

## Analysis

### The Data:
The data set used for this analysis is “Airbnb listings in major US cities” from a Deloitte machine learning competition on Kaggle.com [3]. The dataset consists of 74,111 rows and twenty-nine columns of features for each Airbnb listing. Some of the features include number of bedrooms, type of property, amenities, cancellation policy, reviews, and host information. The feature that will be predicted is log_price. As the prices for this dataset are heavily skewed to the right creating a non-linear relationship between price and the other features that are being examined, the dataset takes the natural logarithm of the price to have log_price follow a normal distribution as shown below.

![image](https://user-images.githubusercontent.com/94664740/226773732-44cdeb89-3f19-4dd4-9fc7-4f1030bc0bd6.png)

There are twenty-eight features in the dataset excluding log_price that will need to be addressed before any models are created. There are 24,392 missing values (na’s) for seven variables and variables that do not offer much information gain so they will be removed as well.

![image](https://user-images.githubusercontent.com/94664740/226773902-be27258c-b200-4803-82d6-76cc27944a5c.png)


The features that were difficult to use, enumerate or proved unnecessary that were removed were id, description, amenities, first_review, host_response_rate, last_review, latitude, longitude, name, and thumbnail_url. This left sixteen feature variables that could be used to predict log_price.
For these sixteen remaining variables there were many missing observations shown below.

![image](https://user-images.githubusercontent.com/94664740/226773928-52a27e58-aaef-4a5b-9eac-40e2e179036b.png)

To manage the missing values for beds, bedrooms, and bathrooms a default of zero was used and for host_has_profile_pic and host_identity_verified a default of False was used to maintain accuracy.

Neighbourhood had 6,872 counts of missing observations and there was not a default value that could be inserted without many hours of research or distorting the dataset, so these rows had to be dropped from the dataset. 

Similarly review_score_rating had many missing observations (16,722), having a missing value here meant that this property had no reviews and would be fairly telling however to protect the relationship between log_price and review_score_rating it was not possible to give them a default of 0 or the mean of the column and these observations were also dropped.

This left 52, 343 rows left for model creation and analysis after dealing with the missing observations.

For the machine learning models to work on this dataset all features needed to be represented numerically. To fix this, all Boolean features such as host_identity_verified, host_has_profile_pic, instant_bookable, and cleaning_fee were transformed from True/False to 1 and 0.

The categorical features room_type, property_type, bed_type, cancellation_policy, city and neighbourhood were all enumerated using Label Encoding to assign a number for each unique type of the feature. Due to the substantial number of features in this dataset it was essential to determine which features were the most influential, a Correlation Matrix, Tree-based Feature Selection and Univariate Feature Selection models were used.

The Univariate model scores each feature based on univariate statistical tests. Using SelectKBest as the scoring program these were the top ten most influential features.

![image](https://user-images.githubusercontent.com/94664740/226774078-6e9b5beb-56c8-4f63-8237-4f302c594292.png)

For the Tree-based feature selection the ExtraTreesClassifier was used to fit a randomized number of decision trees on sub-samples of the dataset to calculate the top performing features.

![image](https://user-images.githubusercontent.com/94664740/226774113-f4ae7103-81c2-4732-ba22-116d92ba3d5c.png)

Lastly the Correlation Matrix heatmap is below, showing a strong relationship between log_price and room type, accommodates, bedrooms and beds.

![image](https://user-images.githubusercontent.com/94664740/226774159-50f5c0a5-e4d6-4d6b-99c3-4204d6ecd004.png)

The scores from each model were considered, and the following features were found to be the top ten important: bed_type, city, instant_bookable, room_type, bedrooms, cancellation_policy, review_scores_rating, bathrooms, beds, accommodates.

Because finding an exact price is not always necessary, determining an optimal price range will be used for these classification models. To do this log_price was discretized into four categories low, medium, high, and very high. To do this both equal width and equal frequency binning methods were used, and the results will be shown for both methods. Equal width bins split the bins so that the width intervals are the same whereas equal frequency makes sure there are an equal number of observations in each bin.

## Decision Tree Models:

First the Decision Tree Classifier was used to predict the bins based on all sixteen remaining features and on only the calculated top ten influential features. For these models’ multiple criteria were used to optimize the model, starting with max depths of 5 and 10 to limit overfitting, as the deeper the tree grows the more complex the model will become and the risk of overfitting the training sample increases. The default ‘best’ splitter was used to choose the best split at each node by evaluating all splits before splitting instead of randomly splitting at each node. Both Gini and Entropy criteria were used where Entropy maximized the information gain at each node and Gini measures the divergences between the probability distributions of the values to equivalent results.

![image](https://user-images.githubusercontent.com/94664740/226774241-07306c0f-b507-4de1-b964-88ff44c9be58.png)

![image](https://user-images.githubusercontent.com/94664740/226774260-8f4c81e3-4eb4-4103-92ae-dfd768cb94bd.png)

## Naive Bayes Model:
Next a Gaussian Naïve Bayes model was used to predict log_price, the advantage being fast processing and solving multiclass problems. This is a classification technique based on the Bayes theorem and the Gaussian model is used to support continuous values and assumes that each class is normally distributed. The algorithm predicts based on occurrences in the dataset so there is the ‘Zero Probability Phenomena’ when there is no class of something in the dataset. This occurred when using equal width bins and the Laplace transformation was used to smooth the results.

![image](https://user-images.githubusercontent.com/94664740/226774459-bf3239ae-6d52-4524-944c-a81f9be21302.png)

## Clustering Models:
K-Means clustering is one of the most popular methods used for classification problems. This model separates the data into clusters based on similar features and common patterns. K-Means Algorithm divides the dataset into k clusters based on similarity and mean distance from the centroid that subgroup formed. The optimal number of clusters was determined using the sum of squared error and elbow plots. 

The elbow plots of all features and the top ten features were almost identical so only one is shown here. 

![image](https://user-images.githubusercontent.com/94664740/226774433-234f22ba-ef6a-4a2a-9c1d-c25e0a137686.png)

Both indicated that 3-4 clusters would perform optimally. A K of 2, 3, 4, 6 and 10 was used in the analysis to show a range of performance.

Hierarchical clustering is another approach to clustering like k-means clustering for identifying groups in the dataset. For this model it is not required to specify the number of clusters to be generated as is required by the k-means approach. Hierarchical clustering results in a tree-based representation of the observations, called a dendrogram that uses Euclidean distance to show clusters.


