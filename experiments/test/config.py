dataset_profiles = {
    "test": {
        "NAME": "test",
        "SEED": 0,
        "X_DIM": 5,
        "Y_DIM": 1,

        "TRAIN_SIZE": 1000,
        "TEST_SIZE": 200,
        "VAL_SIZE": 200,
        "TRAIN_SAMPLE_SPACE": (-10, 10),
        "TEST_SAMPLE_SPACE": (-20, 20),
        "DATA_FUNC": "sum",
        "MU": 0,
        "SIGMA": 3,
    }
}

profiles = {
    "DEFAULT": {
        "NAME": "default",
        "DEVICE": "cpu",
        "SEED": 0,
        "X_DIM": 5,
        "Y_DIM": 1,
        "DATASET": "test",

        "MODEL_TYPE": "BR",
        "HIDDEN_FEATURES": [1024],

        "INFERENCE_TYPE": "svi",
        "SVI_GUIDE": "auto",
        "SVI_ELBO": "elbo",
        "MCMC_KERNEL": "nuts",
        "MCMC_NUM_SAMPLES": 10,
        "MCMC_NUM_WARMUP": 5,
        "MCMC_NUM_CHAINS": 1,
        
        "SAVE_MODEL": True,
        "EPOCHS": 20,
        "LR": 0.001,
        "TRAIN_BATCH_SIZE": 128,
        
        "NUM_X_SAMPLES": 20,
        "NUM_DIST_SAMPLES": 1000,
        "EVAL_BATCH_SIZE": 128,
    }
}