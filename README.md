
LSTM Poetry
===========

This is the result of running a Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)
on a 26MB file of popular song lyrics. The training took about 30h on a GTX 960.
Due to copyright restrictions, the input file is not provided,
but you can download the [trained weights](https://yadi.sk/d/VFv5AuhtqsYrd)
to create some random "poetry".

Simply place the downloaded file in `data/poet` and run `./generate.py`

Change the `TEMPERATURE` parameter in `./generate.py` to vary the degree of randomness.

To train on your own input file, 
 - place the content in the file `input.txt`
 - edit work_dir in `config.py`
 - run `./train.py`


### Installation

You will need [TensorFlow](https://www.tensorflow.org/) (works on version 0.7.1 but not on earlier releases)
and a few other python libs. 
To use the trained example, you don't need special hardware (e.g. a GPU), you can run TensorFlow
on the CPU, but if you plan to train on your own input text, it will take forever without
a GPU-accelerated hardware.

### Examples

Here are some samples of what text can the network produce:

Input text: "A butterfly in "

*A butterfly in* the sun  
Just because I know that I should leave this heart for you  
You said I was falling apart  

I wish I were you  
I wanted you to know how I feel  
I could have settled it all  

It's time to go and do it big and you can be my side  
I can't believe it when I see you  
I'm lost in the world and I can't see you cry  
I'm asking you to love me then let me go  
I can't stop this way  

***

*A butterfly in* the mirror  
The world will find the soul and the truth  
The stars are singing the songs  
I want a holiday, I want you to stay away  
I need you to stay away  

***

*A butterfly in* the middle  
To tell you that the life is safe  

Have you seen the sun and the most of the guy  

We are the spirit of the beast  
The way that we started love  
Let the sun shine on your face  
We will hear you fall apart  




