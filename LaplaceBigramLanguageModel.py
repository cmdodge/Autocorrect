import math, collections

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.types = 0
    self.train(corpus)

#nGramCounts[tuple(['cat', 'in', 'the', 'hat'])] += 1

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    for sentence in corpus.corpus:
        prev = ''
        for datum in sentence.data:  
            word = datum.word
            if word != '<s>':
                if self.unigramCounts[word] == 0:
                    self.types += 1 #V
                self.bigramCounts[tuple([prev, word])] += 1
            self.unigramCounts[word] = self.unigramCounts[word] + 1
            self.total += 1
            prev = word


    #nGramCounts[tuple(['the', 'cat'])] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    prev = ''
    for token in sentence:
        if token != '<s>':
            count = self.bigramCounts[tuple([prev, token])]
            score += math.log(count + 1)
            score -= math.log(self.unigramCounts[prev] + self.types)
        prev = token
    return score
