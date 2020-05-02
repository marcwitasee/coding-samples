'''
CAPP 30122 W'20: Markov models and hash tables

Marc Richardson
'''

import sys
import math
from hash_table import HashTable

HASH_CELLS = 57


class Markov:
    '''Class representing a Markov model for text recognition'''

    def __init__(self, k, s):
        '''
        Construct a new k-order Markov model using the statistics of string "s"
        '''

        self._k = k
        self.uniq_char = len(set(s))
        self.context_table = HashTable(HASH_CELLS, None)

        self.learn(s)


    def log_probability(self, s):
        '''
        Get the log probability of string "s", given the statistics of
        character sequences modeled by this particular Markov model
        This probability is *not* normalized by the length of the string.
        '''

        total = 0.
        prev = s[-self._k:]

        for char in s:
            numerator = 1
            denominator = self.uniq_char
            char_table = self.context_table.lookup(prev)
            if char_table != self.context_table.defval:
                val = char_table.lookup(char)
                denominator += sum(char_table.values())
                numerator += val
            total += math.log(numerator / denominator)
            prev = prev[1:] + char

        return total


    def learn(self, s):
        '''
        Compute the string statistics for the Markov model for a string "s"

        Inputs:
            s (str): text to be processed

        Returns:
            HashTable mapping context sequences to trailing characters
        '''

        prev = s[-self._k:]

        for char in s:
            char_table = self.context_table.lookup(prev)
            if char_table != self.context_table.defval:
                val = char_table.lookup(char) + 1
                char_table.update(char, val)
            else:
                char_table = HashTable(HASH_CELLS, 0)
                char_table.update(char, 1)
                self.context_table.update(prev, char_table)
            prev = prev[1:] + char


def identify_speaker(speaker_a, speaker_b, unknown_speech, k):
    '''
    Given sample text from two speakers, and text from an unidentified speaker,
    return a tuple with the *normalized* log probabilities of each of the
    speakers uttering that text under a "k" order character-based Markov model,
    and a conclusion of which speaker uttered the unidentified text
    based on the two probabilities.
    '''

    s_len = len(unknown_speech)
    model_a = Markov(k, speaker_a)
    model_b = Markov(k, speaker_b)
    log_probability_a = model_a.log_probability(unknown_speech) / s_len
    log_probability_b = model_b.log_probability(unknown_speech) / s_len

    if log_probability_a > log_probability_b:
        prediction = 'A'
    elif log_probability_b > log_probability_a:
        prediction = 'B'
    else:
        prediction = 'B'

    return (log_probability_a, log_probability_b, prediction)


def print_results(res_tuple):
    '''
    Given a tuple from identify_speaker, print formatted results to the screen
    '''
    (likelihood1, likelihood2, conclusion) = res_tuple

    print("Speaker A: " + str(likelihood1))
    print("Speaker B: " + str(likelihood2))

    print("")

    print("Conclusion: Speaker " + conclusion + " is most likely")


def go():
    '''
    Interprets command line arguments and runs the Markov analysis.
    Useful for hand testing.
    '''
    num_args = len(sys.argv)

    if num_args != 5:
        print("usage: python3 " + sys.argv[0] + " <file name for speaker A> " +
              "<file name for speaker B>\n  <file name of text to identify> " +
              "<order>")
        sys.exit(0)

    with open(sys.argv[1], "rU") as file1:
        speech1 = file1.read()

    with open(sys.argv[2], "rU") as file2:
        speech2 = file2.read()

    with open(sys.argv[3], "rU") as file3:
        speech3 = file3.read()

    res_tuple = identify_speaker(speech1, speech2, speech3, int(sys.argv[4]))

    print_results(res_tuple)

if __name__ == "__main__":
    go()
