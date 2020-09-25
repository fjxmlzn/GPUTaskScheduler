# GPU Task Scheduler

**GPU Task Scheduler** is a Python library for scheduling GPU jobs in parallel.

When designing and running neural-network-based algorithms, we often need to test the code on a large set of parameter combinations. To be more efficient, we may also want to distribute the tasks on multiple GPUs in parallel. Writing scripts to achieve these goals may be a headache, especially when the parameter combinations that need to test are complicated, let alone the sophisticated configurations of defining which GPUs to use.

**GPU Task Scheduler** offers you an easy and quick way to do it. All you need is to define the parameter combinations by simple configurations, write the test code in our framework, and then let **GPU Task Scheduler** run the tests in parallel for you. If you already have one test code, don't worry. Migrating your test code to our framework is very easy.

Following are tips on how to install and use it.

## Installation
* **Install from source in editable mode**

  Clone this repo, and run `pip install -e GPUTaskScheduler`.
* **Install by online pip**

  Not available now. Is coming soon.

## Usage

### Configuration
The configuration is defined in a nested Python dictionary structure, which contains multiple optional/mandatory keys. Each key-value pair defines one setting. The definitions of each configuration are as follow.

* **scheduler_config [mandatory]**
  
  The configuration for the scheduler, defined in a Python dictionary. It contains following keys.

	* **gpu [mandatory]**

      This defines the list of GPUs that are available for the scheduler. The value is defined in a Python list, whose values could be (1) GPU IDs like `["0", "1"]`, which means each test requires only one GPU, and the scheduler will distribute tasks among the two GPUs in parallel, (2) lists of GPU IDs like `[["0", "1"],["2", "3"]]`, which means each test may require two GPUs, and the scheduler will distribute tasks among the two two-GPU pairs in parallel. It can also be arbitrary mixed combinations of GPU ID and GPU ID list if you want.
      
    * **result_root_folder [optional, default="results/"]**

      The scheduler will create a folder for each test instance (a test instance is a combination of parameters to test) to store the results. This parameter defines the parent folder in which the result folders are located.

    * **force_rerun [optional, default=False]**
      If force_rerun=False, when the result folder exists, the scheduler will skip the test instance. If force_rerun=True, the scheduler will run all test instances whether or not the result folder exists. 
	
    * **test_config_string_indicator [optional, default="-"]**

      The name of each result folder is in the format of "key1-value1,key2-value2,...", where key*i* is the name of the parameter to test, and value*i* is the value of that parameter. Here "-" is the  test_config_string_indicator. You can customize the indicator by this configuration.

    * **test_config_string_separator [optional, default=","]**
      
      The name of each result folder is in the format of "key1-value1,key2-value2,...", where key*i* is the name of the parameter to test, and value*i* is the value of that parameter. Here "," is the `test_config_string_separator`. You can customize the separator by this configuration.

    * **test_config_string_inst_separator [optional, default="+"]**
    
      The scheduler has an interface to return all test instances in a string. The string for each test instance is the same as the name of result folder, and will be separated by `test_config_string_inst_separator`. You can customize the separator by this configuration.

    * **temp_folder [optional, default="temp/"]**

      The scheduler needs a folder to store some temporary files. This configuration defines the path of the folder. 

    * **log_file [optional, default="worker.log"]**
      
      The console output of each test instance will be stored in a file located in the result folder. The file name is defined by this configuration.

    * **scheduler_log_file_path [optional, default="scheduler.log"]** 

      The scheduler will output a log file, which stores the start time, the end time, and the worker GPU for each test instance. Thi configuration defines the file name of the log file.

    * **config_string_value_maxlen [optional, default=30]**

      The name of each result folder is in the format of "key1-value1,key2-value2,...", where key*i* is the name of the parameter to test, and value*i* is the value of that parameter. If the length of each value is longer than `config_string_value_maxlen`, the value will be truncated to `config_string_value_maxlen`. 

    * **ignored_keys_for_folder_name [optional, default=[]]**

      By default, all keys in the `test_config` section will be used to construct the result folder name. However, the keys listed in `ignored_keys_for_folder_name` will be ignored. 

    * **ignored_characters_for_folder_name [optional, default=[" ", "[", "]"]]**
      
      The characters in `ignored_characters_for_folder_name` will be deleted when constucting the result folder name. The reason for having this option is that TensorFlow does not parse checkpoint paths with certain characters correctly.

* **global_config [optional, default={}]**
  
    It is a Python dictionary, which defines the parameters that are kept the same in all test instances.

* **test_config [optional, default=[{}]]**

    It is a list of Python dictionaries, which defines the parameter combinations to test. In each Python dictionary in the list, it contains multiple key-value pairs. The key is the name of parameter, the value is a list containing all values to test. The "cross product" of those values will be taken to compose the test instances. And the final test instances is the union of all test instances defined by all Python dictionaries in the list. For a clearer explaination, see the example below.

Here is a sample configuration. Assume that it is stored in config.py.

```
config = {
    "scheduler_config": {
        "gpu": ["0", "1", "2"]
    },

    "global_config": {
        "num_run": 5,
        "num_epoch": 200,
    },

    "test_config": [
        {
            "method": ["GAN", "ALI"],
            "num_packing": [1],
            "num_zmode": [1]
        }, 
        {
            "method": ["WGAN"],
            "num_packing": [3],
            "num_zmode": [1, 2]
        }
    ]
}
```
The test instances for this example will be:
```
{"num_run": 5, "num_epoch": 200, "method": "GAN", "num_packing": 1, "num_zmode": 1}
{"num_run": 5, "num_epoch": 200, "method": "ALI", "num_packing": 1, "num_zmode": 1}
{"num_run": 5, "num_epoch": 200, "method": "WGAN", "num_packing": 3, "num_zmode": 1}
{"num_run": 5, "num_epoch": 200, "method": "WGAN", "num_packing": 3, "num_zmode": 2}
```

### Implementing test code interface
The test code should inherit from `gpu_task_scheduler.gpu_task.GPUTask` class, which has two interfaces.

* **def required_env(self) [optional]**

  The scheduler will automatically set GPU-related environment variables (e.g. which GPU to use) using `CUDA_VISIBLE_DEVICES`. But your test code may require some extra environment variable settings. This interface helps you with that. It is called before the test starts running. You can return the required environment variables in a Python dictionary, whose keys are environment variable names and values are the corresponding environment variable values.

* **def main(self) [mandatory]**

  You can write your test code under this function. 

There are two useful class variables accessible in those interfaces:

* **_config**

  It is a Python dictionary, containing the parameter combination for this test. For example,
  ```
  {"num_run": 5, "num_epoch": 200, "method": "GAN", "num_packing": 1, "num_zmode": 1}
  ```

* **_work_dir**

  It is the path of result folder for this test. You can store your results here.

Assume that you implement the class in my_gpu_task.py, and the class name is MyGPUTask.

### Running
Now everything is ready, you can start running the tests by a few lines of code. First let's import the two files you wrote:
```
from config import config
from my_gpu_task import MyGPUTask
```
We also need to import the scheduler class:
```
from gpu_task_scheduler.gpu_task_scheduler import GPUTaskScheduler
```
Now we construct a scheduler by passing config and task class to constructor:
```
scheduler = GPUTaskScheduler(config = config, gpu_task_class = MyGPUTask)
```
and start running the tests:
```
scheduler.start()
```

Now the scheduler will schedule the test instances on the GPUs you set in parallel for you. Whenever a test instance finishes on one GPU, the scheduler will fetch the next test instances and run it on that GPU.

> **WARNING:** If you import theano or tensorflow library in the top module of my_gpu_task.py (or config.py), the code may immediately occupy part of GPU resources before the scheduler starts. Usually that would only waste part of GPU memory resources, but not GPU calculation resources. If you don't want this happen, there are many workarounds:
> * Use [lazy_import](https://github.com/mnmelo/lazy_import) library to import theano or tensorflow, so that they will be loaded at the first real usage.
> * Move import statements to MyGPUTask class body.
> * Implement the test code in another file, and call it in MyGPUTask.main.

> **TODO:** A better way to get around this.

> **WARNING:** When running the test code, the scheduler will automatically set matplotlib's backend to Agg (if matplotlib is installed), because in most cases we don't need to show figures on screen. If this is not what you need, in the begining of your test code's main function, you can use `matplotlib.pyplot.switch_backend` to switch backend, or `reload(matplotlib)` and then choose your desired backend.

> **TODO:** A better way to do this.

### Other useful interfaces

Besides the scheduler class, the library contains another userful class `ConfigManager`, which is used for parsing the configuration file. It is implicitly instantiated in the scheduler class. You can also construct one yourself, as it provides many useful interfaces especially when you want to collect or further process the results.

It contains following public interfaces:

* **def \_\_init\_\_(self, config)**
  The constructor. ``config`` is the configuration object in Python dictionary (same as the one in ``GPUTaskScheduler`` constructor.)

* **def get_all_scheduler_config(self)**
  Returns the scheduler configurations.

* **def get_all_global_config(self)**
  Returns the global configurations.

* **def get_all_test_config(self)**
  Returns the test parameter combinations.

* **def get_all_test_config_in_string(self)**
  Returns all test parameter combinations in a string.

* **def get_next_config(self)**
  Returns the next parameter setting to test.

* **def get_num_left_config(self)**
  Returns the number of untested parameter settings.

## Example
Projects that use this library:
* [PacGAN](https://github.com/fjxmlzn/PacGAN)
* [InfoGAN-CR](https://github.com/fjxmlzn/InfoGAN-CR)
* [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger)

## Contributing
If you find bugs/problems or want to add more features to this library, feel free to submit issues or make pull requests.
