# Naive Bayes Spam Filter
![AUR license](https://img.shields.io/aur/license/s?color=blue&label=license&logo=MIT&logoColor=blue)

Uses Bayes' Theorem to train a model to decide whether an email is spam
or ham depending on the word content within them. We make several adjustments in our model:
1. We consider each email as a set of distinct words
2. We assume that each word in an email is conditionally independent of each other, given that we know whether or not the email is spam (hence naive).
3. We use Laplace smoothing to make our model more robust. Instead of initializing each word count at 0, we start it at 1. This way none of the probabilities of the words will be 0. To account for overestimation, we add 2 to the denominator as well

An example of the output is provided in [example_output.txt](https://github.com/johnsonkuang/NaiveBayesSpamFilter/example_output.txt)

### Built With
- Java

## NaiveBayes.java
The training program for our Naive Bayes model.

1. We iterate over the labelled spam emails, for each word *w* in the entire training set, count how many spam emails contain *w*. Compute *P(w|S)*.
2. Compute *P(w|H)* the same way for ham emails.
3. Compute *P(S)* and *P(H)*.
4. Given an unlabelled email:
    - Create a set of distinct words from the email, ignoring words that are not present in the labelled training data
    - Compute the probability of spam conditioned on the distinct set of words in the unlabelled email.

## SpamFilterMain.java
The runner program for classification.

## License
This project is licensed under the MIT License

