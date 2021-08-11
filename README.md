# bitcoin_prediction
Predict the bitcoin price for tomorrow. 
Provide a sell or buy signal through an api. 
Call it using http://ec2-18-194-140-166.eu-central-1.compute.amazonaws.com/api_v1/predict/bitcoin/
The app use flask as a webservice for the api and a regression model from the scikit-learn module to make the predictions. The app is then located inside a docker image which runs in a aws service.
