
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))

    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    plt.plot(x, running_avg, label='Ego Agent', color='blue')
    plt.title('Running Average of Previous 100 Scores')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid()
    plt.savefig(figure_file)

def plot_learning_curve_multi(x, scores_ego, scores_opp, figure_file):
    running_avg_ego = np.zeros(len(scores_ego))
    running_avg_opp = np.zeros(len(scores_opp))  # separate array

    for i in range(len(running_avg_ego)):
        running_avg_ego[i] = np.mean(scores_ego[max(0, i - 100):(i + 1)])
        running_avg_opp[i] = np.mean(scores_opp[max(0, i - 100):(i + 1)])

    plt.figure()
    plt.plot(x, running_avg_ego, label='Ego Agent', color='blue')
    plt.plot(x, running_avg_opp, label='Opponent Agent', color='red')
    plt.title('Running Average of Previous 100 Scores')
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid()
    plt.savefig(figure_file)
    plt.close()