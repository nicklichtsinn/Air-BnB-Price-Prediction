# Air-BnB-Price-Prediction

## Introduction
Airbnb is an online platform that allows homeowners and renters to put their properties online primarily as vacation rentals and homestays and has seen tremendous growth since its inception in August 2013 in San Francisco California. In December of 2020 Airbnb became a public company with an IPO that raised $3.5 billion, with stocks started the day at $68 a share and ending at $144.71 [1]. With competitors like VRBO and others also joining the niche home sharing market as well as hotel and other options there is a lot of competition to secure quests and keep them.

On Airbnb hosts set their own prices for their listings, although Airbnb and similar home-sharing sites provide recommendations based on a variety of factors, they cannot give an optimal price recommendation for each property that is listed on the platform. There are third-party options for hosts, however these require the hosts to input a base price to give any advice on seasonal or promotional pricing and cut into their bottom line. Airbnb pricing is an important aspect to each host as it can drastically change the number of guests each property has, especially in a crowded market.

Given the considerable number of options that customers have, and that 53% of respondents to the Morgan Stanley study use Airbnb because of the cost savings, this means that over half of Airbnb rentals are happening because of their price [2]. This coupled with the fact that only 11% of listings are reserved by customers on a typical night means that hosts who are not optimizing their pricing strategy are losing out on a lot of money, and there is clearly room for improvement [2]. This paper will attempt to solve this problem by using machine learning techniques with Python to predict the price bin for properties in six major cities in the United States.

## Analysis

### The Data:
The data set used for this analysis is “Airbnb listings in major US cities” from a Deloitte machine learning competition on Kaggle.com [3]. The dataset consists of 74,111 rows and twenty-nine columns of features for each Airbnb listing. Some of the features include number of bedrooms, type of property, amenities, cancellation policy, reviews, and host information. The feature that will be predicted is log_price. As the prices for this dataset are heavily skewed to the right creating a non-linear relationship between price and the other features that are being examined, the dataset takes the natural logarithm of the price to have log_price follow a normal distribution as shown below.

![image](https://user-images.githubusercontent.com/94664740/226773732-44cdeb89-3f19-4dd4-9fc7-4f1030bc0bd6.png)
