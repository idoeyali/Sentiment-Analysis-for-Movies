# Sentiment Analysis for Movie Reviews
Designing networks that predict the viewers’ sentiment (positive/negative) towards the movies they watched based on the review they wrote


## Overview

This project implements various neural network models to perform sentiment analysis on movie reviews. The models include:
- Recurrent Neural Networks (RNN)
- Gated Recurrent Units (GRU)
- Multi-Layer Perceptrons (MLP)
- MLP with Restricted Self-Attention

## Results
This section includes the results of the models, including accuracy and loss on test data.

#### RNN vs GRU
* Review: “If at the beginning I was thinking that the movie do not worth watching it, now i think that everyone should watch it ”
* RNN prediction: Negative review
* GRU prediction: Positive review

Because the RNN model has bad long term memory, the model is wrong and gives prediction of negative review, while the GRU model gives right prediction of positive review.

#### MLP vs MLP+Self-Attention
* Review: ‘this movie is bad, although i was thinking it will be the great’
* MLP prediction: Positive review
* MLP+Self-Attention prediction: Negative review

By the MLP+attention model the word 'bad' has a lot of influence on the word before and after. In the same way the word 'great' has an effect on the words before, but there are no words after so its effect is lower than the word bad on the final review. 
In contrast to the regular MLP prediction where the model gives a score for each word independently to the other words.

## Contributions
Feel free to fork this repository and make your own contributions. Please create a pull request for any significant changes or improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
