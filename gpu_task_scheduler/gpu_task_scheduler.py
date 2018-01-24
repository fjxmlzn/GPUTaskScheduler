from .config_manager import ConfigManager
import multiprocess, time, os, copy, logging, shlex, subprocess, sys
from multiprocess.managers import BaseManager
import os
import pathos
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

DEVNULL = open(os.devnull, 'w')

# code from https://stackoverflow.com/questions/107705/disable-output-buffering
class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
        
def gpu_worker(name, env, _lock, _config, _gpu_task_class, _config_manager, logger):
    while True:
        _lock.acquire()
        config = _config_manager.get_next_config()
        _lock.release()
        if config["config"] == None:
            logger.info("{} finished".format(name))
            break
            
        work_dir = config["work_dir"]
        user_config = config["config"]
        test_config_string = config["test_config_string"]
        
        logger.info("{} receives task {}".format(name, test_config_string))
        
        if os.path.exists(work_dir) and not _config["force_rerun"]:
            logger.info("{} skips task {}".format(name, test_config_string))
            continue
        
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
            
        worker = _gpu_task_class(user_config, work_dir)
        
        new_env = os.environ.copy()
        new_env.update(env)
        new_env.update(worker.required_env())
        pkl_path = os.path.join(_config["temp_folder"], "{}_worker.pkl".format(name))
        with open(pkl_path, "wb") as f:
            pickle.dump(worker, f)
        
        log_file = DEVNULL
        if _config["log_file"] is not None:
            log_file = open(os.path.join(work_dir, _config["log_file"]), "w")
            
        module_path = os.path.abspath(sys.modules[_gpu_task_class.__module__].__file__)
        if module_path[-3:] == "pyc":
            module_path = module_path[:-1]
            
        module_name = _gpu_task_class.__module__
            
        cmd = "start_gpu_task \"{}\" \"{}\" \"{}\" \"{}\"".format(pkl_path, module_name, module_path, os.getcwd())
        args = shlex.split(cmd)
        p = subprocess.Popen(args, env = new_env, stdout = Unbuffered(log_file), stderr = subprocess.STDOUT, cwd = os.getcwd())
        p.wait()
        
        if log_file != DEVNULL:
            log_file.close()
                
        logger.info("{} finishes task {}".format(name, test_config_string))

class GPUTaskScheduler:
    def __init__(self, config, gpu_task_class):
        BaseManager.register('ConfigManager', ConfigManager)
        BaseManager.register('Lock', multiprocess.Lock)
        self._manager = BaseManager()
        self._manager.start()
        self._config_manager = self._manager.ConfigManager(config = config)
        self._lock = self._manager.Lock()
        self._config = self._config_manager.get_all_scheduler_config()
        self._gpu_task_class = gpu_task_class
        if self._config["scheduler_log_file_path"] is not None:
            logging.basicConfig(
                level = logging.INFO,
                format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt = '%a, %d %b %Y %H:%M:%S',
                filename = self._config["scheduler_log_file_path"],
                filemode = 'a'
            )
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)
        else:
            logging.basicConfig(
                level = logging.INFO,
                format = '%(asctime)s %(levelname)s %(message)s',
                datefmt = '%a, %d %b %Y %H:%M:%S',
            )            
        
    def start(self):
        gpu_envs = self._config_manager.get_gpu_envs()
        processes = []
        for gpu_env in gpu_envs:
            try:
                processes.append(pathos.helpers.mp.process.Process(target = gpu_worker, args=(gpu_env["name"], gpu_env["env"], self._lock, self._config, self._gpu_task_class, self._config_manager, logging.getLogger(''))))
            except AttributeError:
                processes.append(pathos.helpers.mp.Process(target = gpu_worker, args=(gpu_env["name"], gpu_env["env"], self._lock, self._config, self._gpu_task_class, self._config_manager, logging.getLogger(''))))                
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        
        
if __name__ == "__main__":
    import config_sample
    import gpu_task
    task_scheduler = GPUTaskScheduler(config = config_sample.config, gpu_task_class = gpu_task.GPUTask)
    task_scheduler.start()
        