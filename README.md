# Sagemaker-sentiments-analysis-deployment
a project to deploy a sentiments analysis model on AWS sagemaker, train a pytorch model endpoint storage it on S3 and deploy it with a lambda function on a website


## Using PyTorch and SageMaker

Project 1 Machine learning engineer ND Deployment

This project is written in Python, it took a dataset of 25.000 movie reviews, those were clean and wrangle and storage on S3 we use a pytorch neural network, and a ml.p2.xlarge instance to train de model, got 85% accuracy a bit less than with XGboost, then we deploy it by creating an endpoint
then we create a rol on IAM with access to sagemaker, and then we create a lambda function with that IAM and on actions we create a REST API just to send request and get a response, on a basic HTML website that when you wrote a review of a movie it will reply to you if it was positivie or negative

By.
Eng. Sergio Beltran
22/04/2020

