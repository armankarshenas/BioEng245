"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 3
"""
import itertools

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.preprocessing import normalize
from tqdm import tqdm
def read_data(file_name, test_size=0.2):
    """
    Q:  read the data from the q2_data.csv.
        the data contains 2 columns with 1 column being the sequence and the second column being
        either 1 or 0. Then, shuffle the data and create a training and testing set based on test_size.

        To read the data, I recommend using pandas' pd.read_csv() function.
        To shuffle and split the data into training/testing sets, I recommend sklearn's train_test_split

        You can use any methods you want to read in the dataset as long as you produce the correct
        output.

        Note, in the case that we are reading q2_data.csv, there should be 5000 sequences.

    Inputs
    - file_name: a string, the name of the data file ("q2_data.csv")
    - test_size: a float to show how many

    Outputs
    - X_train: np.array with shape (N * (1 - test_size)); each item is a sequence from the dataset
    - X_test: np.array with shape (N * test_size)
    - y_train: np.array with shape (N * (1 - test_size)); y[i] is X[i]'s ground-truth label
    - y_test: np.array with shape (N * test_size)
    """
    data = pd.read_csv(file_name, usecols=[0,1], names= ["sequence", "label"],delimiter=" ")
    sequences = np.array(data['sequence'])
    labels = np.array(data['label'])
    X_train,X_test,y_train,y_test = train_test_split(sequences,labels,test_size=test_size,shuffle=True)

    assert type(X_train) == type(X_test) == type(y_train) == type(
        y_test) == np.ndarray, f"read_data() NEEDS to output np.ndarray, instead it's {type(X_train)}!"
    return X_train, X_test, y_train, y_test


def build_transition_matrix(data, k):
    """
    Q:  Here, we will build a transition matrix for a set of sequences.
        Suppose we have a sequence "ACTAGCTACT..." and k = 3, then the
        list of states for this sequence will be:
        ex: states = [ACT, CTA, TAC, ACT,...].

        In order to make it easier for ourselves, we will create a dictionary
        mapping every kmer state to an integer label. So, if our
        kmer2idx dictionary is {'ACT' : 0, 'CTA' : 1, 'TAC' : 2, ...}
        then the states above can be written as [0, 1, 2, 1 ..]

        Next, we build a 2D transition matrix where
        trans_prob[i, j] = probability of state i transitioning to state j.
        ex: trans_prob[3, 1] = probability of state 3 to state 1.

        How do we get this number? Say we want to find trans_prob[1, 2], which is the transition probability
        of 'CTA' to 'TAC'. We count how many transitions from 'CTA' to 'TAC' there are and divide
        this number by the TOTAL number of transitions in the entire dataset, and this gives us
        Pr['CTA' | 'TAC']. However, notice that 'CTA' to 'TAC' in reality is just 'CTAC', we can also consider rewrite this
        as Pr['C' | 'CTA']. Then, we define our transition matrix as a (64, 4) matrix where each row is a possible 3-mer
        and each of the 4 columns is A, G, T, or C. Then, we realize that:
        Pr['A' | 'CTA'] + Pr['T' | 'CTA'] + Pr['G' | 'CTA'] + Pr['C' | 'CTA'] = 1

    Inputs
    - data: np.array with shape (training size); each item is a sequence from the dataset
    - kmer: an int describing the kmer we are interested in

    Outputs
    - kmer2idx: a dictionary that gives us the index of each kmer (ex: kmer2idx['ATC'] = 2 and
                kmer2idx['TCG'] = 4, then trans_mat[2][4] is the probability of "ATCG" happening,
                given that we started with "ATC")
                this is important as it helps us understand what the transition matrix (trans_probs)
                means.
    - trans_probs: np.array with shape (# of states, # of states) with the properties described above
    """

    # Initialize transition probability matrix & generate all possible kmers
    trans_prob = np.zeros((4**k,4))
    nt_array = ["A","G","T","C"]
    kmers = list(itertools.product(nt_array,repeat=k))  # generate a list of all possible kmers combinations, use itertools.product() - read the docs
    kmers = ["".join(item) for item in kmers]
    # build a dictionary to map each kmer to an integer, this will allow us to keep track of where each kmer
    # is located in the transition matrix (trans_prob)
    kmer2idx = {}  # kmer2idx['ATC'] = 4, for example
    for i in range(len(kmers)):
        kmer2idx[kmers[i]] = i

    nt2idx = {'A': 0, 'G': 1, 'T': 2, 'C': 3}  # do not modify this
    print(np.shape(trans_prob),kmer2idx)
    # Iterate through the data to count each transition
    for seq in tqdm(data):
        seq = str(seq)
        seq.strip(" ")
        for i in range(len(seq)-k):
            kmer_local = seq[i:i+k]
            nex_nt = seq[i+k]
            trans_prob[kmer2idx[kmer_local]][nt2idx[nex_nt]] +=1

    trans_prob = trans_prob +1 # apply 1 pseudocount

    # Normalize transition matrix
    trans_prob = normalize(trans_prob, axis = 1, norm = 'l1')

    return kmer2idx, trans_prob
def log_odds_ratio(seq, k, pos_kmer2idx, pos_trans_probs, neg_kmer2idx, neg_trans_probs):
    """
    Q:  this function will calculate the log odds ratio of a sequence with the following formula

    log_odds = log(probability of being class 1 / probability of being class 0)

    Inputs
    - seq: a string, the sequence to be classified
    - k: an int, the kmer substring length
    - pos_kmer2idx: the index system dictionary for the positive (class 1) transition matrix
    - pos_trans_probs: the transition matrix for the positive sequences (class 1 sequences)
    - neg_kmer2idx: same logic as pos_kmer2idx but for the negative sequences (class 0 sequences)
    - neg_trans_probs: same logic as neg_trans_probs but for the negative sequences (class 0 sequences)

    Outputs
    - score: a float, the log odds ratio
    """
    nt2idx = {'A': 0, 'G': 1, 'T': 2, 'C': 3}
    score_pos = 1
    score_neg = 1
    for i in range(len(seq)-k):
        # calculate the log odds ratio using the variables passed
        kmer_local = seq[i:i+k]
        nt_next = seq[i+k]
        prob_local_pos = pos_trans_probs[pos_kmer2idx[kmer_local]][nt2idx[nt_next]]
        score_pos = score_pos*prob_local_pos
        prob_local_neg = neg_trans_probs[neg_kmer2idx[kmer_local]][nt2idx[nt_next]]
        score_neg = score_neg*prob_local_neg
    score = np.log(score_pos/score_neg)
    return score

def classify(seq, k, pos_kmer2idx, pos_trans_probs, neg_kmer2idx, neg_trans_probs):
    """
    Q:  takes a sequence and classifies whether the sequence is a positive class or a negative class.
        if log_odds_ratio > 0, (probability of positive > negative) ==> classify as positive class!
        else if log_odds_ratio < 0 ==> classify as negative class!

    Inputs:
    (check the function above, they have the same specs)

    Outputs:
    - returns an integer, 0 or 1
    """

    # TODO
    score = log_odds_ratio(seq,k,pos_kmer2idx,pos_trans_probs,neg_kmer2idx,neg_trans_probs)
    if score >0:
        return 1
    else:
        return 0

    pass

X_train, X_test, y_train, y_test = read_data("q2_data.csv")
print(np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test))
kmer2idx,trans_prob = build_transition_matrix(X_train,3)

print(kmer2idx)
print(trans_prob)


