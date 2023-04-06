"""Visualisation code for rsnl."""

import jax.numpy as jnp
import jax.random as random
import arviz as az  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.colors as mcolors  # type: ignore


def plot_mcmc(inference_data, folder_name=""):
    az.plot_trace(inference_data, var_names=['~x_adj'], compact=False)
    plt.savefig(f"{folder_name}traceplots.png")
    az.plot_ess(inference_data, var_names=['~x_adj'], kind="evolution")
    plt.savefig(f"{folder_name}ess_plots.png")
    az.plot_autocorr(inference_data, var_names=['~x_adj'])
    plt.savefig(f"{folder_name}autocorr.png")


def plot_theta_posterior(inference_data, reference_values=None, folder_name=""):
    # TODO: MULTIVARIATE
    # TODO: LOOK NICE
    thetas = inference_data.posterior.theta.values
    theta_dims = thetas.shape[-1]
    if theta_dims == 1:
        az.plot_dist(inference_data.posterior.theta.values)
        plt.ylim(bottom=0)
        plt.xlabel("Theta", fontsize=25)  # TODO! USE TEX FOR PROPER PLOTS
        plt.ylabel("Density", fontsize=25)
        if reference_values is not None:
            plt.axvline(reference_values['theta_1'], color='red', linestyle='dashed',
                        linewidth=2)
    else:
        theta_plot = {}
        var_name_map = {}
        if reference_values is not None:
            reference_values = {}
        for i in range(theta_dims):
            theta_plot['theta_' + str(i+1)] = thetas[:, i]
        for ii, k in enumerate(theta_plot):
            # var_name_map[k] = fr'$\{k[:-1]}_{k[-1]}$'
            var_name_map[k] = 'theta_' + str(ii+1)  # TODO: for now...
            if reference_values is not None:
                reference_values[var_name_map[k]] = reference_values[ii]  # why does ref_vals match labels and not data? ah well
        fig, axes = plt.subplots(theta_dims, theta_dims,
                                 sharey=False, figsize=(16, 16))
        axes = az.plot_pair(theta_plot,
                            kind='kde',
                            reference_values=reference_values,
                            reference_values_kwargs={'color': 'red',
                                                     'marker': 'X',
                                                     'markersize': 12},
                            kde_kwargs={'hdi_probs': [0.05, 0.25, 0.5, 0.75, 0.95],
                                        'contour_kwargs': {"colors": None, "cmap": plt.cm.viridis},
                                        'contourf_kwargs': {"alpha": 0}},
                            ax=axes,
                            labeller=az.labels.MapLabeller(var_name_map=var_name_map),
                            textsize=18,
                            marginals=True,
                            marginal_kwargs={'label': 'RSNL'},
                            # show=False
                            # figsize=(64, 64)
                            )

    plt.savefig(f"{folder_name}theta_posterior.pdf", bbox_inches='tight')
    plt.clf()


def plot_adj_posterior(inference_data, folder_name=""):
    # plt.rcParams['text.usetex'] = True  # TODO: latex
    plt.rcParams.update({'font.size': 25})

    # Generate prior samples
    # laplace_mean = jnp.zeros(10)
    # laplace_var = 0.3 * jnp.abs(x_obs_standard)
    # TODO: SMARTER
    rng_key = random.PRNGKey(0)
    summary_dims = inference_data.posterior.adj_params.values.shape[-1]
    prior_samples = random.laplace(rng_key, shape=(10000, summary_dims))
    for i in range(summary_dims):  # TODO: lazy
        az.plot_dist(inference_data.posterior.adj_params.values[:, :, i].flatten(),  # TODO: CHANGED
                     label='Posterior',
                     color='black')
        az.plot_dist(prior_samples[:, i],
                     color=mcolors.CSS4_COLORS['limegreen'],
                     plot_kwargs={'linestyle': 'dashed'},
                     label='Prior')
        # subscript = i + 1
        # plt.xlabel("$\gamma_{%s}$" % (i+1), fontsize=30)  # TODO: latex
        plt.xlabel(f"gamma_{i+1}$", fontsize=25)
        plt.ylabel("Density", fontsize=25)
        plt.ylim(bottom=0)
        plt.legend(fontsize=20)
        # plt.title("$b_0 = 0.01$")
        plt.show()
        plt.savefig(f'{folder_name}adj_param{i+1}.pdf', bbox_inches='tight')
        plt.clf()


def plot_and_save_all(inference_data, true_params, folder_name=""):
    theta_plot = {}
    var_name_map = {}
    # adj_plot = {}
    # adj_name_map = {}

    # TODO! TEMP
    theta_dims = inference_data.posterior.theta.values.shape[-1]
    # summary_dims = inference_data.posterior.adj_params.values.shape[-1]

    for i in range(theta_dims):
        key_name = 'theta_' + str(i+1)
        theta_plot[key_name] = inference_data.posterior.theta.values.flatten()
        var_name_map[key_name] = f'theta_{i+1}'

    # for i in range(summary_dims):
    #     key_name = 'adj param' + str(i+1)
    #     adj_plot[key_name] = inference_data.posterior.adj_params.values[:, :, i].flatten()
    #     var_name_map[key_name] = fr'$\gamma_{i+1}$'

    var_name_map = {}
    reference_values = {}
    for ii, k in enumerate(theta_plot):
        # var_name_map[k] = fr'$\{k[:-1]}_{k[-1]}$'
        reference_values[k] = true_params[ii]  # why does ref_vals match labels and not data? ah well

    plot_theta_posterior(inference_data, folder_name=folder_name,
                         reference_values=reference_values)
    plot_adj_posterior(inference_data, folder_name=folder_name)
    plot_mcmc(inference_data, folder_name=folder_name)
