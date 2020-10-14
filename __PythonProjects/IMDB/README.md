IMDB Sentiment Analysis - End2End
---

Background: (title)

The IMDb dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative. 
The dataset contains an even number of positive and negative reviews.

Goal: (title)

This project example can be used in order to demonstrate end-to-end ML pipeline (flow) in cnvrg.io platform.

Files: (title)

    - Put the file ```IMDB_original.csv``` in cnvrg data set.
    - Put the files:
 
        (1) ```imdb_prep.py``` - pre-processing of the ```IMDB_original.csv``` file. It outputs a file called ```IMDB_processed.csv``` (the capital letters are case sensitive!),
    and a file called ```words_and_values.json``` which is a dictionary describes the numeric value of each word after the pre-processing.
    
        (2) ```rnn.py``` - the neural network model. The file gets the output file (processed data set) via the field ```--data```,
        number of epochs training as ```--epochs``` and batch size  ```--batch_size``` in training. The file outputs a model file called ```model.h5```.
    
        (3) ```predict.py``` - this file is used for deploying an Endpoint. The file uses the two files: ```model.h5``` and ```words_and_values.json```.
    
        in the files directory of the project.
        
        
Construction of the end-to-end flow: (title)

In order to have an end-to-end flow, we first need to have the files which predict demands in the files directory of the project.
So, construct the following structre:

[imdb_ds] --> [imdb_prep.py (--data="/data/imdb_ds/IMDB_original.csv")] --> [rnn.py (--data="IMDB_processed.csv" --epochs=X --batch_size=Y)]

This structure produces the two files which is required in order to launch serving (Endpoint), merge the last commit to master.

Now, you can launch serving (Endpoint) - use the predict.py file and the predict method.

In order to have the full end-to-end flow, run the pipeline described above with the small edition:

--> [EndPoint]

and run again.

Now, you are able to use the serving :-)
  

requirements: (title)

    - sklearn
    - cnvrg
    - tensorflow >= 2.0.0
    - pandas
    - json

