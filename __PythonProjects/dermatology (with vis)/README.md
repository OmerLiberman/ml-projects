Dermatology - Skin Cancer Prediction - End2End
---

Background: (title)

****add some content here ***

Goal: (title)

This project example can be used in order to demonstrate end-to-end ML pipeline (flow) in cnvrg.io platform.

Files: (title)

    - Put the file ```derm_ds.csv``` in cnvrg data set.
    - Put the files:
 
        (1) ```model.py``` - This file includes the short pre-processing, short training of xgboost (xgboost.XGBClassifier) model and saving the model.
        The file recevies the param ```--data``` (suppose to recevie /data/derm_ds/derm_ds.csv) and saves a model called 'model.sav'.
    
        (2) ```predict.py``` - In order to perform prediction, the user needs to deliver numbers to the predict function.
        The model requires 34 parameters in order to perform prediction, but don't worry - If the user delivers less numbers, The vector is padded 
        with zeros. Then, the file loads the trained model ("model.sav") and returns the prediction (number in [1, 6]).        
        
Construction of the end-to-end flow: (title)

In order to have an end-to-end flow, we first need to have the files which predict demands in the files directory of the project.
So, construct the following structre:

[derm_ds] --> [model.py (--data="/data/derm_ds/derm_ds.csv")]

This structure produces the model which is required in order to launch serving (Endpoint), merge the last commit to master.

Now, you can launch serving (Endpoint) - use the predict.py file and the predict method.

In order to have the full end-to-end flow, run the pipeline described above with the small edition:

--> [EndPoint]

and run again.

Now, you are able to use the serving :-)
  

requirements: (title)

    - xgboost
    - pandas
    - sklearn
    - pandas

