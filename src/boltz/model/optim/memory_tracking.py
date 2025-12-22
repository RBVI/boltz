import gc
import datetime
import inspect

import torch
import numpy as np

dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, detail=True, path='', verbose=False, device=0):
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f'memory-use-{datetime.datetime.now():%d-%b-%y-%H:%M:%S}.txt'
        self.verbose = verbose
        self.begin = True
        self.device = device

    def get_tensors(self):
        tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]
        return tensors

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        if torch.cuda.is_available():
            bytes = torch.cuda.memory_allocated()
        elif torch.backends.mps.is_available():
            bytes = torch.mps.current_allocated_memory()
        else:
            bytes = 0
        return bytes / 1024**2

    def get_cache_size(self):
        if torch.cuda.is_available():
            bytes = torch.cuda.memory_reserved()
        elif torch.backends.mps.is_available():
            bytes = torch.mps.driver_allocated_memory()
        return bytes / 1024**2

    def clear_cache(self, log = True):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if log:
            with open(self.gpu_profile_fn, 'a+') as f:
                f.write(f"\n{datetime.datetime.now():%d-%b-%y-%H:%M:%S}\nCleared torch cache\n")

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def track(self, comment = ''):
        """
        Track the GPU memory usage
        """
        frameinfo = inspect.stack(context=2)[2]
        where_str = f'{comment}: {frameinfo.filename} line {frameinfo.lineno} in {frameinfo.function}'

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                        f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")
                self.begin = False

            f.write(f"\n{datetime.datetime.now():%d-%b-%y-%H:%M:%S}\n"
                    f"{where_str}\n"
                    f"Total Tensor Memory: {self.get_tensor_usage():>7.1f}Mb\n"
                    f"Allocated Tensor Memory: {self.get_allocate_usage():>7.1f}Mb\n"
                    f"Total Cache Memory: {self.get_cache_size():>7.1f}Mb\n\n")

            if self.print_detail is True:
                tensors = self.get_tensors()
                ts_list = [(tensor.size(), tensor.dtype) for tensor in tensors]
                new_tensor_sizes = {(type(x).__name__,
                                     tuple(x.size()),
                                     ts_list.count((x.size(), x.dtype)),
                                     np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                     str(x.dtype).split('.')[-1],
                                     str(x.device)) for x in tensors}

                new_tensors = list(new_tensor_sizes - self.last_tensor_sizes)
                new_tensors.sort(key = lambda t: t[2]*t[3], reverse = True)
                for t, s, n, m, data_type, dev in new_tensors:
                    f.write(f'+ | {str(n):>5} * {str(s):<20} | {(m*n):<6.1f} M | {str(t):<10} | {data_type:>8} | {dev.split(":")[0]}\n')

                old_tensors = list(self.last_tensor_sizes - new_tensor_sizes)
                old_tensors.sort(key = lambda t: t[2]*t[3], reverse = True)
                for t, s, n, m, data_type, dev in old_tensors:
                    f.write(f'- | {str(n):>5} * {str(s):<20} | {(m*n):<6.1f} M | {str(t):<10} | {data_type:>8} | {dev.split(":")[0]}\n')

                self.last_tensor_sizes = new_tensor_sizes


_mem_tracker = None
def mem_track_start():
    global _mem_tracker
    if _mem_tracker is None:
        _mem_tracker = MemTracker()
    return _mem_tracker

def mem_track(comment = ''):
    if _mem_tracker:
        _mem_tracker.track(comment)

def mem_clear_cache(log = True):
    if _mem_tracker:
        _mem_tracker.clear_cache(log = log)
