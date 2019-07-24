import matplotlib.pyplot as plt

def plot_simple(x_y_pairs, pairs_labels, x_label, y_label, filename, x_lim=None, y_lim=None, epochs=False):
    
    if epochs:
        plt.figure(figsize=(50, 8))
    else:
        plt.figure(figsize=(10,10))

    linestyles = ['-', '--', '-.', ':']
    for pair, label, linestyle in zip(x_y_pairs, pairs_labels, linestyles):
        plt.plot(pair[0], pair[1], linestyle, label=label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if x_lim:
        plt.xlim(x_lim)
    if y_lim:
        plt.ylim(y_lim)
    plt.legend()
    plt.savefig(filename)
    plt.clf()
