import numpy as np 

class MNNaiveBayes:
    """
    This class is written with the standard python library and NumPy.
    The goal is to recreate a multinomial Naive Bayes classifcation model from scratch.
    
    Usage:
    1. Instantiate the model
    2. Fit it on a corpus of documents and an array of their binary classifications
    3. Predict on another corpus to get an array of binary classes
    """
  def __init__(self, k=0.5):
    self.k = k
    self.cat0_count = 0
    self.cat1_count = 0
    self.total_count = self.cat0_count + self.cat1_count
    self.cat_0_prior = 0
    self.cat_1_prior = 0
    self.cat_0_prior, self.cat_1_prior
    self.word_probs = []
    self.vocab = []

  def tokenize(self, document):
    """
    Take in a document and return a list of words
    """
    doc = document.lower()
    # remove non-alpha characters
    stop_chars = '''0123456789!()-[]{};:'"\,<>./?@#$%^&*_~'''
    
    tokens = ""
    # iterate through and make each token
    for char in doc:
      if char not in stop_chars:
        tokens += char

    return tokens.split() # now a list of tokens
  
  def count_words(self, X, y):
    """
    X is an array of documents
    y is an array of targets, 0 or 1
    Output a dictionary of {word: (cat0_count, cat1_count)...}
    """
    counts = {}
    # need to figure our this loop, want to iterate over both of them, I see why it was paired before
    for document, category in zip(X, y):
        for token in self.tokenize(document):
          # Initialize a dict entry with 0 counts
          if token not in counts:
            counts[token] = [0,0]
          # Now that it exists, add to the category count for that word
          counts[token][category] += 1
    return counts

  def prior_prob(self, counts):
    
    # Iterate through counts dict and add up each word count by category
    cat0_word_count = cat1_word_count = 0
    for word, (cat0_count, cat1_count) in counts.items():
        cat0_word_count += cat0_count
        cat1_word_count += cat1_count

    # save attributes to the class
    self.cat0_count = cat0_word_count
    self.cat1_count = cat1_word_count
    self.total_count = self.cat0_count + self.cat1_count

    # Get the prior prob by dividing words in each cat by total words
    cat_0_prior = cat0_word_count / self.total_count
    cat_1_prior = cat1_word_count / self.total_count
    return cat_0_prior, cat_1_prior

  def word_probabilities(self, counts):
    """turn the word_counts into a list of triplets
    word, p(w | cat0), and p(w | cat1)"""
    # Here we apply the smoothing term, self.k, so that words that aren't in
    # the category don't get calculated as 0
    self.vocab = [word for word, (cat0, cat1) in counts.items()]
    return [(word,
    (cat0 + self.k) / (self.cat0_count + 2 * self.k),
    (cat1 + self.k) / (self.cat1_count + 2 * self.k))
    for word, (cat0, cat1) in counts.items()]

  def fit(self, X, y):
    # Take all these functions and establish probabilities of input
    counts = self.count_words(X, y)
    self.cat_0_prior, self.cat_1_prior = self.prior_prob(counts)
    self.word_probs = self.word_probabilities(counts)

  def predict(self, test_corpus):
    # Split the text into tokens,
    # For each category: calculate the probability of each word in that cat
    # find the product of all of them and the prior prob of that cat
    y_pred = []
    for document in test_corpus:
      # Every document get their own prediction probability
      log_prob_cat0 = log_prob_cat1 = 0.0
      tokens = self.tokenize(document)
        # Iterate through the training vocabulary and add any log probs that match
        # if no match don't do anything. We just need a score for each category/doc
      for word, prob_cat0, prob_cat1 in self.word_probs:
        if word in tokens:
          # Because of 'overflow' best to add the log probs together and exp
          log_prob_cat0 += np.log(prob_cat0)
          log_prob_cat1 += np.log(prob_cat1)
        # get each of the category predictions including the prior
      cat_0_pred = self.cat_0_prior * np.exp(log_prob_cat0)
      cat_1_pred = self.cat_1_prior * np.exp(log_prob_cat1)
      if cat_0_pred >= cat_1_pred:
        y_pred.append(0)
      else:
        y_pred.append(1)
    return y_pred
      
