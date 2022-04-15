from matplotlib import pyplot as plt
import numpy as np

def svm_results(OUTPUT_FOLDER, kinds, mean_scores, scores_std):
    print("FINAL RESULTS")
    print("kinds: ", kinds)
    print("mean_scores: ", mean_scores)
    print("scores_std: ", scores_std, "\n")

    plt.figure(figsize=(6, 4))
    positions = np.arange(len(kinds)) * .1 + .1
    plt.barh(positions, mean_scores, align='center', height=.05, xerr=scores_std)
    yticks = [k.replace(' ', '\n') for k in kinds]
    plt.yticks(positions, yticks)
    plt.gca().grid(True)
    plt.gca().set_axisbelow(True)
    plt.gca().axvline(.8, color='red', linestyle='--')
    plt.xlabel('Classification accuracy\n(red line = chance level)')
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER + "clustering_results.png")
