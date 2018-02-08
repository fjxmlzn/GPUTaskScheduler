config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2"]
    }

    "global_config": {
        "num_run": 5,
        "num_epoch": 200,
    },

    "test_config": [
        {
            "method": ["GAN", "ALI"],
            "num_packing": [1, 2],
            "num_zmode": [1, 2]
        }, 
        {
            "method": ["WGAN"],
            "num_packing": [3, 4],
            "num_zmode": [1, 2]
        }
    ]
}