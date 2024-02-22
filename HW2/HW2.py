"""
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 2
"""

"""Q 1.1"""
def rev_complement(seq):
    """
    Q:  find the reverse complement of the sequence.
        note, assume seq contains all capital letters and are restricted
        to A, T, G, and C.
    Ex:
    >>> find_complement(seq="ATGTACTAGCTA")
    TAGCTAGTACAT

    Input
    - seq: the sequence to find the complement for

    Output
    - comp: the complement
    """
    seq_output = ""
    for i in range(len(seq)):
        ch = seq[len(seq)-i-1]
        if ch == "A":
            seq_output = seq_output + "T"
        elif ch == "T":
            seq_output = seq_output + "A"
        elif ch == "C":
            seq_output = seq_output + "G"
        elif ch == "G":
            seq_output = seq_output + "C"
    return  seq_output


"""Q 1.2"""
def gc_content(seq):
    """
    Q:  find the GC% of the sequence.
    Ex:
    >>> gc_content("ATCGACTCGAGTCGTACGTTCACG")
    0.5416666666666666

    Input
    - seq: sequence

    Output
    - gc = GC% (a float between 0 and 1)
    """
    gc = 0
    for i in range(len(seq)):
        if seq[i] == "C" or seq[i] == "G":
            gc +=1
    return gc/len(seq)


"""Q 1.3"""
def find_motif_freq(motif, seq):
    """
    Q:  given a target motif, find its frequency in a dna strand.
    Ex:
    >>> find_motif_freq("AA", "AAAAAAAAA")
    8
    >>> find_motif_freq("ATC", "ACTGACTATCGTCAGTCGATCTAATCCTG")
    3

    Input
    - motif: target substring
    - seq: sequence to search in

    Output
    - freq: frequency of the given motif
    """
    num_motifs = 0
    n = len(seq)
    m = len(motif)
    for i in range(n-m+1):
        if seq[i:i+m] == motif:
            num_motifs += 1
    return num_motifs

"""Q 1.4"""
def find_binding_site(bind, seq):
    """
    Q:  given a sequence, find the first position of the binding site.
        note, the binding site is the reverse complement of bind.
        hint: you can call the rev_complement() method earlier.
        hint: return -1 if the binding site is NOT found in seq
    Ex:
    >>> find_binding_site("ATGC", "ACTCGACTCAGCATCATACGGACTC")
    10

    Inputs
    - bind: the short binding-sequence that will bind to the sequence seq
    - seq: sequence to be bind to

    Outputs
    - pos: position (0 indexed)
    """
    rev_com = rev_complement(bind)
    if find_motif_freq(rev_com,seq) == 0:
        return -1
    else:
        for i in range(len(seq)-len(bind)+1):
            if seq[i:i+len(bind)] == rev_com:
                return i



if __name__ == '__main__':
    # print(rev_complement("ATGTACTAGCTA"))
    # print(gc_content("ATCGACTCGAGTCGTACGTTCACG"))
    # print(find_motif_freq("ATC", "ACTGACTATCGTCAGTCGATCTAATCCTG"))
    # print(find_binding_site("ATGC", "ACTCGACTCAGCATCATACGGACTC"))
    pass

  """
  School: University of California, Berkeley
  Course: BIOENG 145/245
  Author: Yorick Chern
  Instructor: Liana Lareau
  Assignment 2
  """

import numpy as np

"""Q 2.1"""
def num_die_sum(num_die, total, trials=1000000):
    """
    Q:  given n fair 6-sided die, what is the probability that we roll a certain sum?

    Inputs
    - num_die: number of fair 6-sided (ranges 1-6) die to be thrown
    - total: the sum we are looking for
    - trials: number of simulations ran

    Output
    - prob: probability that the sum of num_die rolled = total
    """


    # check that total > minimum sum or < maximum sum ==> otherwise
    # there is no chance of happening ==> probability = 0.00
    if total < num_die:
        return 0
    if total > num_die * 6:
        return 0

    # use np.random.randint(...) to generate a matrix with the size of (# of trials, # of die)
    # where each row is a trial and each column is the value of a dice throw
    rolls = np.random.randint(1,7,[trials,num_die])

    # sum up each row to obtain the sum of each trial, use np.sum()
    sums = np.sum(rolls,axis=-1)

    # how many elements in sums = the sum we are looking for?
    tally = np.sum(sums==total)
    prob = tally / trials
    return prob

"""Q 2.2"""
def correct_papers(num_papers, trials=1000000):
    """
    Q:  a professor got mad at his students and throws a pile of n papers on the
        floor and asks each student to pick up a random paper from the floor.
        On average, how many students get their own paper back?

    Input
    - num_papers: number of papers to be thrown

    Output
    - avg: the average number of students who got their own paper
    """
    sum = 0
    # use list comprehension or np.linspace() to generate a list from 1 to num_papers (or 0 to num_papers - 1)
    papers = list(range(0,num_papers))
    for i in range(trials):
        # use np.random.permutation()
        shuffled_papers = np.random.permutation(papers)
        sum += np.sum(shuffled_papers == papers)
    avg = sum / trials
    return avg


"""Q 2.3"""
def monte_carlo_pi(num_points):
    """
    Q:  estimate pi using num_points

    HINT:
    1.  generate n random (x, y) points
    2.  calculate the number of (x, y) points that falls within a unit circle
    3.  divide this number by the total number of points generated
    4.  multiple this ratio by the area of square that bounds the unit circle (what does this ratio represent?)
    5.  use this number to determine pi

    Input
    - num_points: number of points generated randomly

    Output
    - pi: estiamted pi
    """

    # generate pairs of (x, y) coordinates within the range of (-1, 1)
    # use np.random.rand()
    x, y = np.random.rand(num_points), np.random.rand(num_points)

    # follow the algorithm above
    inside_circle = 0
    for i in range(num_points):
        if x[i]**2 + y[i]**2 <=1:
            inside_circle += 1
    avg_points = inside_circle/num_points
    pi = 4*avg_points
    return pi



"""Q 2.4"""
def roll_until_repeat(n_sided, trials=1000000):
    """
    Q:  on average, how many rolls do we need until we see 2 consecutive rolls of the same value?
        Ex: 2, 4, 1, 5, 3, 6, 4, 4 ==> we see two 4's in a row after 6 rolls

    Input
    - n_sided: a fair n-sided dice

    Output
    - avg: average number of rolls needed
    """

    total_rolls = 0
    num_rolls_per_trial = 50    # this is the number of rolls
    for i in range(trials):
        # start by using np.random.randint(...) using the num_rolls_per_trials
        rolls = np.random.randint(1,n_sided,num_rolls_per_trial)
        l = 0
        while l <num_rolls_per_trial-1:
            if rolls[l] == rolls[l+1]:
                total_rolls += l
                break
            else:
                l+=1

    avg = total_rolls / trials
    return avg




if __name__ == '__main__':
	# some test cases for you to follow
    # print(num_die_sum(2, 4))              # 0.08319
    # print(correct_papers(1000))           # 1.000
    # print(monte_carlo_pi(1000000))        # 3.14
    # print(roll_until_repeat(6, 10000))    # 5
    pass

      """
School: University of California, Berkeley
Course: BIOENG 145/245
Author: Yorick Chern
Instructor: Liana Lareau
Assignment 2
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets,model_selection,metrics
from scipy.stats import multivariate_normal

# some helper functions - do not change anything here
def pdf(x, mean, cov):
    return multivariate_normal.pdf(x, mean, cov)

def make_data(n_samples, features, k_classes, sep=2.0):
    centers = np.random.rand(k_classes, features)
    for i in range(k_classes):
        centers[i] += sep * i
    stds = np.ones_like(centers)
    X, y = datasets.make_blobs(
        n_samples=n_samples,
        n_features=features,
        centers=centers,
        cluster_std=stds
    )
    return X, y

class GaussianDiscriminator:

    def __init__(self):
        self.gaussians = []     # stores the (class_label, mean, covariance) tuples

    def fit(self, X, y, num_classes):
        """
        This function fits a Gaussian distribution to each class in the dataset.
	    These Gaussian distributions are appended to self.gaussians as a tuple consisting of (class_label, mean, covariance)

        Inputs
        - X: the data, numpy array with shape (n, d) = (sample size, feature dims)
        - y: the class labels, numpy array with shape (n, ) and integers from 0, ..., num_classes-1
        - num_classes: the number of classes, int

        Outputs
        - None
        """

        # iterate through each class and find its mean and covariance matrix
        # HINT: use np.mean, np.cov, and np.where.
        # save the class k, the mean of the class sample, and covariance matrix of the class sample as a tuple
        # and append it to self.gaussians
        # self.gaussians.append((k, mean, cov))
        for i in range(num_classes):
            idx_cond = y == i
            X_cond = X[idx_cond,:]
            mean_local = np.mean(X_cond,axis=0)
            cov_matrix_local = np.cov(np.transpose(X_cond))
            self.gaussians.append((i,mean_local,cov_matrix_local))

    def predict(self, x):
        """
        This function will classify the dataset x, returning a predicted label for each datapoint.

        Inputs
        - x: data to be predicted, numpy array with shape (n, d) = (sample size, feature dims)

        Outputs
        - pred: np.array with shape (n, ), where each item is the classification of each data
        """
        pred = []
        # iterate through each data
        for i in range(x.shape[0]):
            # for each data iterate through each gaussian in self.gaussian
            # find the pdf using the helper function provided above
            # HINT: remember, in Python, tuples can be decomposed like so
            # a, b, c = (a, b, c)
            prob_local = []
            for k in range(len(self.gaussians)):
                class_idx, mean_local, cov_local = self.gaussians[k]
                pdf_local = pdf(x[i, :], mean_local, cov_local)
                prob_local.append(pdf_local)
            prob_local = np.array(prob_local)
            pred.append(np.argmax(prob_local))
        return np.array(pred)


if __name__ == '__main__':

	# do not make any modifications here, it's a simple visualization for your implementation.
    X, y = make_data(1200, 2, 2, sep=3.0)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    idx = np.where(y_train == 0)[0]
    plt.scatter(X_train[idx, 0], X_train[idx, 1], label='0', color='r')
    idx = np.where(y_train == 1)[0]
    plt.scatter(X_train[idx, 0], X_train[idx, 1], label='1', color='b')
    plt.legend()
    plt.title("Train data")
    plt.show()  # click on "x" to exit out the graph and continue running the program

    gd = GaussianDiscriminator()
    gd.fit(X_train, y_train, 2)

    y_pred = gd.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))


    idx = np.where(y_test == 0)[0]
    plt.scatter(X_test[idx, 0], X_test[idx, 1], label='0', color='r')
    idx = np.where(y_test == 1)[0]
    plt.scatter(X_test[idx, 0], X_test[idx, 1], label='1', color='b')
    plt.legend()
    plt.title("Test data & predictions")
    plt.show()
