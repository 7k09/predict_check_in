# predict_check_in

Predict which business a user is checking into based on their location, accuracy, and timestamp. 
More about the data:
https://www.kaggle.com/c/facebook-v-predicting-check-ins/data

train.csv, test.csv have the following columns:
row_id: id of the check-in event
x y: coordinates
accuracy: location accuracy 
time: timestamp
place_id: id of the business, this is the target of prediction
