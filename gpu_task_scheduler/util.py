from contextlib import contextmanager
import os, sys

@contextmanager
def environ(env):
    original_environ = os.environ.copy()
    os.environ.update(env)
    yield
    os.environ = original_environ
    
    
@contextmanager
def redirect_output(stdout, stderr):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if stdout is not None:
        sys.stdout = stdout
    if stderr is not None:
        sys.stderr = stderr
    yield
    sys.stdout = original_stdout
    sys.stderr = original_stderr