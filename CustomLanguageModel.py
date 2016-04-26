import math, collections

class CustomLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.trigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.types = 0
    self.train(corpus)

#Trigram Laplace Smoothing

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
        w_1 = '<s>'
        w_2 = '<s>'
        for datum in sentence.data:  
            w_3 = datum.word
            if w_3 != '<s>':
                if self.unigramCounts[w_3] == 0:
                    self.types += 1 #V
                self.bigramCounts[tuple([w_2, w_3])] += 1
                self.trigramCounts[tuple([w_1, w_2, w_3])] += 1
            self.unigramCounts[w_3] = self.unigramCounts[w_3] + 1
            self.total += 1
            w_1 = w_2
            w_2 = w_3

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    w_1 = '<s>'
    w_2 = '<s>'
    for token in sentence:
        if token != '<s>':
            count = self.trigramCounts[tuple([w_1, w_2, token])]
            if count > 0:
                score += math.log(count)
                score -= math.log(self.bigramCounts[tuple([w_2, token])])
            elif self.bigramCounts[tuple([w_2, token])] > 0: 
                score += math.log(self.bigramCounts[tuple([w_2, token])])
                score -= math.log(self.unigramCounts[w_2])
            else: 
                score += math.log(self.unigramCounts[token] + 1)
                score -= math.log(self.total + self.types)

        w_1 = w_2
        w_2 = token
    return score