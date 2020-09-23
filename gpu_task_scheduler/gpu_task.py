class GPUTask:
    def __init__(self, config, work_dir):
        self._config = config
        self._work_dir = work_dir

    def required_env(self):
        return {}

    def main(self):
        pass
