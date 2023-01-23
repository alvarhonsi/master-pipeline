import seaborn as sns
import matplotlib.pyplot as plt



def plot_distribution(samples, save_path=None, ax=None):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    sns.kdeplot(samples, fill=True, ax=ax)

    if ax is None:
        ax.set_ylabel("Density")
        ax.set_xlabel("y")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")

    if save_path:
        plt.savefig(save_path)

def plot_comparison(data_sample, post_sample, save_path=None, ax=None):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    sns.kdeplot(data_sample, fill=True, ax=ax, label="Data")
    sns.kdeplot(post_sample, fill=True, ax=ax, label="Posterior")
    

    if ax is None:
        ax.set_ylabel("Density")
        ax.set_xlabel("y")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")
    
    if save_path:
        plt.savefig(save_path)

"""Plots a grid of comparisons between the posterior and data distributions"""
def plot_comparison_grid(data_samples, posterior_samples, grid_size=2, save_path=None):
    assert posterior_samples.shape[1] == data_samples.shape[1]
    num_x = data_samples.shape[1]
    assert grid_size**2 <= num_x

    print(data_samples.shape)
    print(posterior_samples.shape)

    sns.set_style("darkgrid")
    sns.set_context("paper")

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10,10))
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle("Posterior vs Data Distributions", fontsize=15)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)
    fig.text(0.5, 0.04, 'Y', ha='center', fontsize=15)
    fig.text(0.04, 0.5, 'Density', va='center', rotation='vertical', fontsize=15)

    for i, ax in enumerate(axs):
        plot_comparison(data_samples[:,i], posterior_samples[:,i], ax=ax)
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')    

    if save_path:
        plt.savefig(save_path)