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
        rolls = np.random.randint(1,n_sided+1,num_rolls_per_trial)
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
