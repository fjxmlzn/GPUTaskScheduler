import itertools
import copy
import collections
import os
import sys


class ConfigManager:
    def __init__(self, config):
        self._config = config
        self._scheduler_config = None
        self._test_config = None
        self._global_config = None
        self._counter = 0

        self._parse_scheduler_config()
        self._parse_global_config()
        self._parse_test_config()

    def _parse_scheduler_config(self):
        if "scheduler_config" in self._config:
            self._scheduler_config = self._config["scheduler_config"]

            self._scheduler_config.setdefault("force_rerun", False)
            self._scheduler_config.setdefault(
                "result_root_folder", os.path.abspath("results/"))
            self._scheduler_config.setdefault(
                "test_config_string_separator", ",")
            self._scheduler_config.setdefault(
                "test_config_string_indicator", "-")
            self._scheduler_config.setdefault(
                "test_config_string_inst_separator", "+")
            self._scheduler_config.setdefault(
                "temp_folder", os.path.abspath("temp/"))
            if not os.path.exists(self._scheduler_config["temp_folder"]):
                os.makedirs(self._scheduler_config["temp_folder"])
            self._scheduler_config.setdefault("log_file", "worker.log")
            self._scheduler_config.setdefault(
                "scheduler_log_file_path", "scheduler.log")

            if "gpu" not in self._scheduler_config:
                raise ValueError("could not find {} in {}".format(
                    "gpu", "scheduler_config"))
            if len(self._scheduler_config["gpu"]) < 1:
                raise ValueError(
                    "please specify at least one gpu in {}".format(
                        "scheduler_config"))
            for i, gpu in enumerate(self._scheduler_config["gpu"]):
                if type(gpu) == list:
                    self._scheduler_config["gpu"][i] = ",".join(gpu)
                elif type(gpu) != str:
                    raise ValueError(
                        "gpu can only be a string or a list of strings")
        else:
            raise ValueError("could not find {} in {}".format(
                "scheduler_config", "config"))

    def _parse_global_config(self):
        if "global_config" in self._config:
            self._global_config = self._config["global_config"]
        else:
            self._global_config = {}

    def _parse_test_config(self):
        if "test_config" in self._config:
            self._test_config = []
            for inst in self._config["test_config"]:
                keys = []
                values = []
                for config_key in inst:
                    keys.append(config_key)
                    values.append(inst[config_key])
                for pairs in itertools.product(*values):
                    obj = {}
                    for i in range(len(keys)):
                        obj[keys[i]] = pairs[i]
                    self._test_config.append(obj)
        else:
            self._test_config = [{}]

    def _get_test_config_string(self, config):
        ans = ""
        for key in sorted(config.keys()):
            ans += "{}{}{}{}".format(
                key, self._scheduler_config["test_config_string_indicator"],
                config[key],
                self._scheduler_config["test_config_string_separator"])
        return ans

    def get_all_scheduler_config(self):
        return self._scheduler_config

    def get_all_global_config(self):
        return self._global_config

    def get_all_test_config(self):
        return self._test_config

    def get_all_test_config_in_string(self):
        ans = ""
        for inst in self._config["test_config"]:
            for key in sorted(inst.keys()):
                ans += "{}{}{}{}".format(
                    key,
                    self._scheduler_config["test_config_string_indicator"],
                    inst[key],
                    self._scheduler_config["test_config_string_separator"])
            ans += self._scheduler_config["test_config_string_inst_separator"]
        return ans

    def get_gpu_envs(self):
        envs = []
        for gpu in self._scheduler_config["gpu"]:
            obj = {}
            obj["name"] = "gpu{}".format(gpu)
            obj["env"] = {}
            obj["env"]["CUDA_VISIBLE_DEVICES"] = gpu
            envs.append(obj)
        return envs

    def get_next_config(self, create_dir=False):
        ans = {
            "config": None
        }
        if self._counter < len(self._test_config):
            ans["config"] = copy.deepcopy(self._global_config)
            ans["config"].update(self._test_config[self._counter])
            ans["test_config_string"] = self._get_test_config_string(
                self._test_config[self._counter])
            ans["work_dir"] = os.path.join(
                self._scheduler_config["result_root_folder"],
                ans["test_config_string"])
            self._counter += 1
        return ans

    def get_num_left_config(self):
        return len(self._test_config) - self._counter

    def get_num_gpu(self):
        return len(self._scheduler_config["gpu"])


if __name__ == "__main__":
    import config_sample
    config_manager = ConfigManager(config_sample.config)
    print(config_manager.get_all_scheduler_config())
    print(config_manager.get_all_global_config())
    print(config_manager.get_all_test_config())
    print(config_manager.get_gpu_envs())

    for i in range(config_manager.get_num_left_config()):
        print(config_manager.get_next_config())
