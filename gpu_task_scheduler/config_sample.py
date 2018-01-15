config = {
    "scheduler_config": {
        "backend": "theano",
        "gpu": ["gpu0", "gpu1", "gpu2"],
        "force_rerun": False,
    },

    "theano_config": {
        "theanorc_template_file": "sample.theanorc"
    },

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