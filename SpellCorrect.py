import math
from Datum import Datum
from Sentence import Sentence
from HolbrookCorpus import HolbrookCorpus
from UniformLanguageModel import UniformLanguageModel
from UnigramLanguageModel import UnigramLanguageModel
from StupidBackoffLanguageModel import StupidBackoffLanguageModel
from LaplaceUnigramLanguageModel import LaplaceUnigramLanguageModel
from LaplaceBigramLanguageModel import LaplaceBigramLanguageModel
from CustomLanguageModel import CustomLanguageModel
from EditModel import EditModel
from SpellingResult import SpellingResult
import types

# Modified version of Peter Norvig's spelling corrector
"""Spelling Corrector.

Copyright 2007 Peter Norvig. 
Open source code under MIT license: http://www.opensource.org/licenses/mit-license.php
"""

import re, collections

class SpellCorrect:
  """Spelling corrector for sentences. Holds edit model, language model and the corpus."""

  def __init__(self, lm, corpus):
    self.languageModel = lm
    self.editModel = EditModel('../data/count_1edit.txt', corpus)

  def correctSentence(self, sentence):
    """Assuming exactly one error per sentence, returns the most probable corrected sentence.
       Sentence is a list of words."""

    if len(sentence) == 0:
      return []

    bestSentence = sentence[:] #copy of sentence
    bestScore = float('-inf')
    

      # TODO: select the maximum probability sentence here, according to the noisy channel model.
      # Tip: self.editModel.editProbabilities(word) gives edits and log-probabilities according to your edit model.
      #      You should iterate through these values instead of enumerating all edits.
      #       self.editModel.editProbabilities(word) returns list of tuples of (word, log probability)
      # Tip: self.languageModel.score(trialSentence) gives log-probability of a sentence

#For example, if the sentence is " I need to thw my car", the probability of the edit (thw-> the) 
#may be higher than the probability of the edit (the->tow), but the probability of the sentence 
#"I need to tow my car" would be higher than "I need to the my car".
 
#You're calculating the score of the sentence by adding the logprob of the corrected sentence to the logprob of the edit. 
    

    for i in xrange(0, len(sentence)):
        for (word, probability) in self.editModel.editProbabilities(sentence[i]):
          temp = sentence[:]
          temp[i] = word
          if probability + self.languageModel.score(temp) >= bestScore:
            bestSentence = temp
            bestScore = probability + self.languageModel.score(temp)

    return bestSentence


  def evaluate(self, corpus):  
    """Tests this speller on a corpus, returns a SpellingResult"""
    numCorrect = 0
    numTotal = 0
    testData = corpus.generateTestCases()
    for sentence in testData:
      if sentence.isEmpty():
        continue
      errorSentence = sentence.getErrorSentence()
      hypothesis = self.correctSentence(errorSentence)
      if sentence.isCorrection(hypothesis):
        numCorrect += 1
      numTotal += 1
    return SpellingResult(numCorrect, numTotal)

  def correctCorpus(self, corpus): 
    """Corrects a whole corpus, returns a JSON representation of the output."""
    string_list = [] # we will join these with commas,  bookended with []
    sentences = corpus.corpus
    for sentence in sentences:
      uncorrected = sentence.getErrorSentence()
      corrected = self.correctSentence(uncorrected)
      word_list = '["%s"]' % '","'.join(corrected)
      string_list.append(word_list)
    output = '[%s]' % ','.join(string_list)
    return output

def main():
  """Trains all of the language models and tests them on the dev data. Change devPath if you
     wish to do things like test on the training data."""
  trainPath = '../data/holbrook-tagged-train.dat'
  trainingCorpus = HolbrookCorpus(trainPath)

  devPath = '../data/holbrook-tagged-dev.dat'
  devCorpus = HolbrookCorpus(devPath)

  print 'Unigram Language Model: ' 
  unigramLM = UnigramLanguageModel(trainingCorpus)
  unigramSpell = SpellCorrect(unigramLM, trainingCorpus)
  unigramOutcome = unigramSpell.evaluate(devCorpus)
  print str(unigramOutcome)

  print 'Uniform Language Model: '
  uniformLM = UniformLanguageModel(trainingCorpus)
  uniformSpell = SpellCorrect(uniformLM, trainingCorpus)
  uniformOutcome = uniformSpell.evaluate(devCorpus) 
  print str(uniformOutcome)

  print 'Laplace Unigram Language Model: ' 
  laplaceUnigramLM = LaplaceUnigramLanguageModel(trainingCorpus)
  laplaceUnigramSpell = SpellCorrect(laplaceUnigramLM, trainingCorpus)
  laplaceUnigramOutcome = laplaceUnigramSpell.evaluate(devCorpus)
  print str(laplaceUnigramOutcome)

  print 'Laplace Bigram Language Model: '
  laplaceBigramLM = LaplaceBigramLanguageModel(trainingCorpus)
  laplaceBigramSpell = SpellCorrect(laplaceBigramLM, trainingCorpus)
  laplaceBigramOutcome = laplaceBigramSpell.evaluate(devCorpus)
  print str(laplaceBigramOutcome)

  print 'Stupid Backoff Language Model: '  
  sbLM = StupidBackoffLanguageModel(trainingCorpus)
  sbSpell = SpellCorrect(sbLM, trainingCorpus)
  sbOutcome = sbSpell.evaluate(devCorpus)
  print str(sbOutcome)

  print 'Custom Language Model: '
  customLM = CustomLanguageModel(trainingCorpus)
  customSpell = SpellCorrect(customLM, trainingCorpus)
  customOutcome = customSpell.evaluate(devCorpus)
  print str(customOutcome)

if __name__ == "__main__":
    main()
