## SAM inference 

This repository contains two methods for running inference for the SAM model:

- Through running a flask server on a bare metal GPU
- By using modal

The files **sam.py** and **app.py** have the code for running inference and the code for flask server respectively.\
The file **modal.py** has the code for creating a modal App, downloading a model and running it's inference.\
(NOTE: one needs to run ```modal deploy``` before trying to run inference. Check out the documentation for more)
