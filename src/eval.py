import datetime
import time
import json
import os
import pyro
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import train
import random
from modules.distributions import PredictivePosterior, DataDistribution, NormalPosterior
from modules.metrics import difference_mean, difference_std, KL_divergance_normal
from modules.plots import plot_comparison, plot_comparison_grid
from modules.datageneration import load_data, data_functions
from pipeline_util import draw_data_samples, save_bnn, load_bnn


def sample_predictive_posterior(bnn, x_sample, num_samples=100):
    return bnn.sample_predictive(x_sample, num_predictions=num_samples).squeeze(-1).cpu().detach().numpy()

def get_pred_dists(bnn, dataloader, num_predictions=1000, device="cpu"):
    results = {}
    # Evaluate error
    mean_all, std_all = None, None
    batch_num = 0
    for num_batch, (input_data, observation_data) in enumerate(iter(dataloader), 1):
        input_data, observation_data = input_data.to(
            device), observation_data.to(device)
        mean, std = bnn.predict(input_data, num_predictions=num_predictions)
        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()
        
        if batch_num == 0:
            mean_all = mean
            std_all = std
        else:
            mean_all = np.vstack((mean_all, mean))
            std_all = np.vstack((std_all, std))

        batch_num+=1

    results["mean"] = mean_all.reshape(-1).tolist()
    results["std"] = std_all.reshape(-1).tolist()

    return results


def evaluate_error(bnn, dataloader, num_predictions=100, device="cpu"):
    results = {}
    # Evaluate error
    rmse, loglikelihood, mae = 0, 0, 0
    batch_num = 0
    for num_batch, (input_data, observation_data) in enumerate(iter(dataloader), 1):
        input_data, observation_data = input_data.to(
            device), observation_data.to(device)
        mse, ll, absolute_err = bnn.get_error_metrics(
            input_data, observation_data, num_predictions=num_predictions, reduction="mean")
        rmse += mse
        loglikelihood += ll
        mae += absolute_err
        batch_num = num_batch
    rmse = (rmse / batch_num).sqrt()
    loglikelihood = loglikelihood / batch_num
    mae = mae / batch_num

    results["rmse"] = rmse.item()
    results["loglikelihood"] = loglikelihood.item()
    results["mae"] = mae.item()

    return results


def evaluate_posterior(posterior_samples, data_samples):
    results = {}

    post_std_is_zero = any(np.std(posterior_samples, axis=0) == 0)
    data_std_is_zero = any(np.std(data_samples, axis=0) == 0)

    # Mean kl divergence
    if not post_std_is_zero and not data_std_is_zero:
        kl_div = np.mean(KL_divergance_normal(posterior_samples, data_samples))
        results["kl_div_to_data"] = np.float64(kl_div)
    else:
        results["kl_div_to_data"] = np.float64(-1)
    # Difference in mean
    mean_diff = np.mean(difference_mean(posterior_samples, data_samples))
    results["mean_diff_to_data"] = np.float64(mean_diff)

    # Difference in standard deviation
    std_diff = np.mean(difference_std(posterior_samples, data_samples))
    results["std_diff_to_data"] = np.float64(std_diff)

    return results


def eval(config, dataset_config, DIR, bnn=None, device=None, reruns=1):
    NAME = config["NAME"]
    SEED = config.getint("SEED")
    DEVICE = device if device != None else config["DEVICE"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")
    DATASET = config["DATASET"]

    OBS_MODEL = config["OBS_MODEL"]
    INFERENCE_TYPE = config["INFERENCE_TYPE"]

    DATA_FUNC = dataset_config["DATA_FUNC"]
    MU = dataset_config.getfloat("MU")
    SIGMA = dataset_config.getfloat("SIGMA")

    NUM_X_SAMPLES = config.getint("NUM_X_SAMPLES")
    NUM_DIST_SAMPLES = config.getint("NUM_DIST_SAMPLES")
    EVAL_BATCH_SIZE = config.getint("EVAL_BATCH_SIZE")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load data
    (x_train, y_train), _, (x_test_in_domain, y_test_in_domain), (x_test_out_domain,
                                                                  y_test_out_domain) = load_data(f"{DIR}/datasets/{DATASET}", load_val=False)
    x_test = torch.vstack((x_test_in_domain, x_test_out_domain))
    y_test = torch.vstack((y_test_in_domain, y_test_out_domain))

    # Make dataset
    train_dataset = TensorDataset(x_train, y_train)
    test_in_domain_dataset = TensorDataset(x_test_in_domain, y_test_in_domain)
    test_out_domain_dataset = TensorDataset(
        x_test_out_domain, y_test_out_domain)
    test_dataset = TensorDataset(x_test, y_test)

    # Make dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)
    test_in_domain_dataloader = DataLoader(
        test_in_domain_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)
    test_out_domain_dataloader = DataLoader(
        test_out_domain_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True)

    # Ready samples directory
    if not os.path.exists(f"{DIR}/results/{NAME}/posterior-samples"):
        os.mkdir(f"{DIR}/results/{NAME}/posterior-samples")

    if not os.path.exists(f"{DIR}/results/{NAME}/data-samples"):
        os.mkdir(f"{DIR}/results/{NAME}/data-samples")

    if not os.path.exists(f"{DIR}/results/{NAME}/sanity-checks"):
        os.mkdir(f"{DIR}/results/{NAME}/sanity-checks")

    # samp_x: (NUM_X_SAMPLES, X_DIM), samp_y: (NUM_X_SAMPLES)
    train_x_sample, _ = draw_data_samples(
        train_dataloader, NUM_X_SAMPLES, device=DEVICE)
    test_x_sample, _ = draw_data_samples(
        test_dataloader, NUM_X_SAMPLES, device=DEVICE)
    test_in_domain_x_sample, _ = draw_data_samples(
        test_in_domain_dataloader, NUM_X_SAMPLES, device=DEVICE)
    test_out_domain_x_sample, _ = draw_data_samples(
        test_out_domain_dataloader, NUM_X_SAMPLES, device=DEVICE)

    np.save(f"{DIR}/results/{NAME}/data-samples/train_x.npy",
            train_x_sample.cpu().detach().numpy())
    np.save(f"{DIR}/results/{NAME}/data-samples/test_x.npy",
            test_x_sample.cpu().detach().numpy())
    np.save(f"{DIR}/results/{NAME}/data-samples/test_in_domain_x.npy",
            test_in_domain_x_sample.cpu().detach().numpy())
    np.save(f"{DIR}/results/{NAME}/data-samples/test_out_domain_x.npy",
            test_out_domain_x_sample.cpu().detach().numpy())

    # Sample true distribution from data
    # data_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    func = data_functions[DATA_FUNC]
    data_train_dist = DataDistribution(func, MU, SIGMA, train_x_sample)
    data_test_dist = DataDistribution(func, MU, SIGMA, test_x_sample)
    data_in_domain_dist = DataDistribution(
        func, MU, SIGMA, test_in_domain_x_sample)
    data_out_domain_dist = DataDistribution(
        func, MU, SIGMA, test_out_domain_x_sample)

    data_train_samples = data_train_dist.sample(
        NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_test_samples = data_test_dist.sample(
        NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_in_domain_samples = data_in_domain_dist.sample(
        NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_out_domain_samples = data_out_domain_dist.sample(
        NUM_DIST_SAMPLES).cpu().detach().numpy()

    np.save(f"{DIR}/results/{NAME}/data-samples/train_dist_samples.npy",
            data_train_samples)
    np.save(f"{DIR}/results/{NAME}/data-samples/test_dist_samples.npy",
            data_test_samples)
    np.save(f"{DIR}/results/{NAME}/data-samples/test_in_domain_dist_samples.npy",
            data_in_domain_samples)
    np.save(f"{DIR}/results/{NAME}/data-samples/test_out_domain_dist_samples.npy",
            data_out_domain_samples)

    print("data samples: ", data_in_domain_samples.shape)

    for run in range(1, reruns + 1):
        #####
        # Load model
        bnn = train.make_inference_model(
            config, dataset_config, device=DEVICE)
        bnn = load_bnn(bnn, config,
                             load_path=f"{DIR}/models/{NAME}/checkpoint_{run}.pt", device=DEVICE)

        #####

        # Ready results directory
        if not os.path.exists(f"{DIR}/results/{NAME}"):
            os.mkdir(f"{DIR}/results/{NAME}")

        start = time.time()
        print(f"using device: {DEVICE}")
        print(f"====== evaluating profile {NAME} - {run} ======")

        # Draw samples from relevant distributions

        pred_train_samples = sample_predictive_posterior(
            bnn, train_x_sample, num_samples=NUM_DIST_SAMPLES)
        pred_test_samples = sample_predictive_posterior(
            bnn, test_x_sample, num_samples=NUM_DIST_SAMPLES)
        pred_in_domain_samples = sample_predictive_posterior(
            bnn, test_in_domain_x_sample, num_samples=NUM_DIST_SAMPLES)
        pred_out_domain_samples = sample_predictive_posterior(
            bnn, test_out_domain_x_sample, num_samples=NUM_DIST_SAMPLES)

        np.save(f"{DIR}/results/{NAME}/posterior-samples/train_samples_{run}.npy",
                pred_train_samples)
        np.save(f"{DIR}/results/{NAME}/posterior-samples/test_samples_{run}.npy",
                pred_test_samples)
        np.save(f"{DIR}/results/{NAME}/posterior-samples/test_in_domain_samples_{run}.npy",
                pred_in_domain_samples)
        np.save(f"{DIR}/results/{NAME}/posterior-samples/test_out_domain_samples_{run}.npy",
                pred_out_domain_samples)

        print("pred samples: ", pred_train_samples.shape)

        # Sanity Checks
        plot_comparison_grid(pred_train_samples, data_train_samples, grid_size=(3, 3), figsize=(20, 20), x_samples=train_x_sample,
                             kl_div=True, title="Posterior samples - Train", plot_mean=True, save_path=f"{DIR}/results/{NAME}/sanity-checks/train_sanity_{run}.png")
        plot_comparison_grid(pred_in_domain_samples, data_in_domain_samples, x_samples=test_in_domain_x_sample, grid_size=(3, 3), figsize=(
            20, 20), kl_div=True, title="Posterior samples - In Domain", plot_mean=True, save_path=f"{DIR}/results/{NAME}/sanity-checks/test_in_domain_sanity_{run}.png")
        plot_comparison_grid(pred_out_domain_samples, data_out_domain_samples, x_samples=test_out_domain_x_sample, grid_size=(3, 3), figsize=(
            20, 20), kl_div=True, title="Posterior samples - Out of Domain", plot_mean=True, save_path=f"{DIR}/results/{NAME}/sanity-checks/test_out_domain_sanity_{run}.png")

        # Evaluate

        results = {}
        cases = ["train", "test", "in_domain", "out_domain"]
        dataloaders = [train_dataloader, test_dataloader,
                       test_in_domain_dataloader, test_out_domain_dataloader]
        pred_samples = [pred_train_samples, pred_test_samples,
                        pred_in_domain_samples, pred_out_domain_samples]
        data_samples = [data_train_samples, data_test_samples,
                        data_in_domain_samples, data_out_domain_samples]
        eval_cases = zip(cases, dataloaders, pred_samples, data_samples)
        for case, dataloader, pred_sample, data_sample in eval_cases:
            results[case] = {}
            print(f"Evaluating {case}...")
            error = evaluate_error(
                bnn, dataloader, num_predictions=1000, device=DEVICE)
            results[case]["error"] = error

            eval_posterior = evaluate_posterior(
                pred_sample, data_sample)
            results[case]["predictive_eval"] = eval_posterior

        with open(f"{DIR}/results/{NAME}/results_{run}.json", "w") as f:
            json.dump(results, f, indent=4)


        uncertainties = {}
        cases = ["train", "in_domain", "out_domain"]
        dataloaders = [train_dataloader,
                       test_in_domain_dataloader, test_out_domain_dataloader]
        pred_samples = [pred_train_samples,
                        pred_in_domain_samples, pred_out_domain_samples]
        data_samples = [data_train_samples,
                        data_in_domain_samples, data_out_domain_samples]
        eval_cases = zip(cases, dataloaders, pred_samples, data_samples)
        for case, dataloader, pred_sample, data_sample in eval_cases:
            uncertainties[case] = {}
            print(f"Evaluating Uncertainty in {case}...")
            pred_dist = get_pred_dists(
                bnn, dataloader, num_predictions=1000, device=DEVICE)
            uncertainties[case]["pred_dist"] = pred_dist
            uncertainties[case]["mean_predictive_scale"] = np.mean(pred_dist["std"])
            uncertainties[case]["min_predictive_scale"] = np.min(pred_dist["std"])
            uncertainties[case]["max_predictive_scale"] = np.max(pred_dist["std"])

        with open(f"{DIR}/results/{NAME}/predictive_uncertainties_{run}.json", "w") as f:
            json.dump(uncertainties, f, indent=4)

        print("Saving weight distributions...")
        weight_data = {}
        weight_dist = bnn.get_weight_distributions()
        weight_data["sites"] = {k: v.reshape(-1).cpu().tolist() for k, v in weight_dist.items()}
        
        #get mean weight scale
        weight_scale_list = []
        for name, data in weight_data["sites"].items():
            if "scale" in name:
                weight_scale_list.extend(data)
        weight_data["mean_weight_scale"] = np.mean(weight_scale_list)
        weight_data["min_weight_scale"] = np.min(weight_scale_list)
        weight_data["max_weight_scale"] = np.max(weight_scale_list)
        print("mean weight scale: ", weight_data["mean_weight_scale"])



        with open(f"{DIR}/results/{NAME}/weight_data_{run}.json", "w") as f:
            json.dump(weight_data, f, indent=4)


        # Get time and format to HH:MM:SS
        elapsed_time = str(datetime.timedelta(seconds=time.time() - start))
        print(f"Eval done in {elapsed_time}")
