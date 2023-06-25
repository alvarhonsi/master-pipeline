import seaborn as sns
import matplotlib.pyplot as plt
from .metrics import KL_divergance_normal, difference_mean, difference_std


def lineplot(data, x_label=None, y_label=None, figsize=(10, 5), ax=None, save_path=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(data=data, ax=ax)

    if ax is None:
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")

    if save_path:
        plt.savefig(save_path)


def plot_elbo(train_stats, save_path=None, figsize=(20, 8)):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle("ELBO for epochs and pr. minibatch", fontsize=15)

    lineplot(train_stats["elbo_minibatch"], ax=ax[0])
    ax[0].set_ylabel("ELBO")
    ax[0].set_xlabel("Minibatch")

    lineplot(train_stats["elbo_epoch"], ax=ax[1])
    ax[1].set_ylabel("ELBO")
    ax[1].set_xlabel("Epoch")

    if save_path:
        plt.savefig(save_path)


def plot_rmse(train_stats, save_path=None, figsize=(20, 8)):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle("RMSE for epochs and pr. minibatch", fontsize=15)

    lineplot(train_stats["rmse_minibatch"], ax=ax[0])
    ax[0].set_ylabel("RMSE")
    ax[0].set_xlabel("Minibatch")

    lineplot(train_stats["rmse_epoch"], ax=ax[1])
    ax[1].set_ylabel("RMSE")
    ax[1].set_xlabel("Epoch")

    if save_path:
        plt.savefig(save_path)


def plot_distribution(samples, save_path=None, ax=None, figsize=(10, 10)):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(samples, fill=True, ax=ax)

    if ax is None:
        ax.set_ylabel("Density")
        ax.set_xlabel("y")
    else:
        ax.set_ylabel("")
        ax.set_xlabel("")

    if save_path:
        plt.savefig(save_path)


def plot_comparison(post_sample, data_sample, x_sample=None, x_label="y", y_label="Density", title=None, save_path=None, ax=None, figsize=(10, 10), kl_div=False, plot_mean=False):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    data_std = data_sample.std()
    post_std = post_sample.std()

    title = ""

    if x_sample is not None:
        title += f"X: {[round(x.item(), 2) for x in x_sample]} \n"

    if kl_div:
        kl = KL_divergance_normal(
            post_sample, data_sample) if data_std > 1e-5 and post_std > 1e-5 else -1
        mean_diff = difference_mean(post_sample, data_sample)
        std_diff = difference_std(post_sample, data_sample)
        title += f"KL-divergence: {kl:.4f} | Mean diff: {mean_diff:.4f} | Std diff: {std_diff:.4f}"

    standalone = False
    if ax is None:
        standalone = True

    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    if data_std > 1e-5:
        sns.kdeplot(data_sample, fill=True, ax=ax, label="Data")
    if post_std > 1e-5:
        sns.kdeplot(post_sample, fill=True, ax=ax, label="Posterior")
    if plot_mean:
        ax.axvline(data_sample.mean(), color="red",
                   label="Data mean", alpha=0.4)
        ax.axvline(post_sample.mean(), color="green",
                   label="Posterior mean", alpha=0.4)
    ax.legend()
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if title:
        ax.set_title(title, wrap=True)

    if save_path:
        plt.savefig(save_path)


"""Plots a grid of comparisons between the posterior and data distributions"""


def plot_comparison_grid(posterior_samples, data_samples, x_samples=None, title=None, grid_size=(2, 2), save_path=None, figsize=(10, 10), kl_div=False, plot_mean=False):
    assert posterior_samples.shape[1] == data_samples.shape[1]
    num_x = data_samples.shape[1]
    assert (grid_size[0] * grid_size[1]) <= num_x

    sns.set_style("darkgrid")
    sns.set_context("paper")

    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1,
                        right=0.9, hspace=0.4, wspace=0.4)
    fig.text(0.5, 0.04, 'Y', ha='center', fontsize=15)
    fig.text(0.04, 0.5, 'Density', va='center',
             rotation='vertical', fontsize=15)

    for i, ax in enumerate(axs):
        x_samp = x_samples[i] if x_samples is not None else None
        plot_comparison(posterior_samples[:, i], data_samples[:, i],
                        x_sample=x_samp,  kl_div=kl_div, plot_mean=plot_mean, ax=ax)

    if save_path:
        plt.savefig(save_path)

    plt.close()

# post_samples: array of samples from different posteriors


def plot_comparisons(post_samples, data_sample, labels, x_sample=None, x_label="y", y_label="Density", title=None, save_path=None, ax=None, figsize=(10, 10), kl_div=False, plot_mean=False):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    data_std = data_sample.std()

    title = ""

    if x_sample is not None:
        title += f"X: {[round(x.item(), 2) for x in x_sample]} \n"

    standalone = False
    if ax is None:
        standalone = True

    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    if data_std > 1e-5:
        sns.kdeplot(data_sample, fill=True, ax=ax, label="Data")

    for i, post_sample in enumerate(post_samples):
        post_std = post_sample.std()
        if post_std > 1e-5:
            sns.kdeplot(post_sample, fill=False, ax=ax, label=labels[i])
        else:
            ax.axvline(post_sample.mean(), color="green",
                        label=labels[i], alpha=0.4)

    ax.legend()
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    if title:
        ax.set_title(title, wrap=True)

    if save_path:
        plt.savefig(save_path)

# post_samples: array of samples from different posteriors


def plot_comparisons_grid(posterior_samples, data_samples, x_samples=None, title=None, grid_size=(2, 2), save_path=None, figsize=(10, 10), kl_div=False, plot_mean=False):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axs = axs.flatten()
    fig.tight_layout()
    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1,
                        right=0.9, hspace=0.4, wspace=0.4)
    fig.text(0.5, 0.04, 'Y', ha='center', fontsize=15)
    fig.text(0.04, 0.5, 'Density', va='center',
             rotation='vertical', fontsize=15)

    for i, ax in enumerate(axs):
        x_samp = x_samples[i] if x_samples is not None else None
        plot_comparisons(
            posterior_samples[i], data_samples[:, i], x_sample=x_samp, ax=ax)

    if save_path:
        plt.savefig(save_path)
