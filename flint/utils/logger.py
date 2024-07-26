import sys
import os

class Logger(object):
    def __init__(self, fpath=None, state=None):
        self.state = state
        self.console = sys.stdout
        self.file = None
        if fpath is not None and state:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()
        
    def info(self, obj):
        if self.state:
            content = '{}\n'.format(obj)
            self.console.write(content)
            if self.file is not None:
                self.file.write(content)
                
    def write(self, key, obj):
        if self.state:
            content = '{}: {}\n'.format(key, obj)
            self.console.write(content)
            if self.file is not None:
                self.file.write(content)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # self.console.close()
        if self.file is not None:
            self.file.close()