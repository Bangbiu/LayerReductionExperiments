import os
import torch
from model import *
import torchprof
from torch.profiler import profile, record_function, ProfilerActivity

# Directory
os.chdir(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_eval_autograd(model_name, inputData):
    print("Start Eval：", model_name)

    model = eval(model_name)(num_classes=5).to(device)
    # load model weights
    weights_path = "./weights/{}.pth".format(model_name)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    print("using {} device.".format(device))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(inputData)
    print(prof)

def infer_eval_torchprof(model_name, inputData):
    print("Start Eval：", model_name)

    model = eval(model_name)(num_classes=5).to(device)
    # load model weights
    weights_path = "./weights/{}.pth".format(model_name)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    print("using {} device.".format(device))

    with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
        for i in range(1000):
            model(inputData)

    # equivalent to `print(prof)` and `print(prof.display())`
    print(prof.display(show_events=False))
    """
    with profile(activities=[ProfilerActivity.CPU], 
        record_shapes=True,
        with_stack=True,
        with_modules =True
        ) as prof:
        with record_function("model_forward"):
            for i in range(1,100):
                model(inputData)

    print(prof.key_averages(group_by_input_shape=True).table())
    """
    """
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=5, # during this phase profiler is not active
            warmup=2, # during this phase profiler starts tracing, but the results are discarded
            active=6, # during this phase profiler traces and records data
            repeat=2), # specifies an upper bound on the number of cycles
        with_stack=True # enable stack tracing, adds extra profiling overhead
    ) as prof:
        model(inputData)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    """

if __name__ == '__main__':
    models_list = ["AlexNet_ConcatenateConv1to2", "AlexNet_ConcatenateConv1to2", "AlexNet_ConcatenateConv1to3", "AlexNet_ConcatenateConv1to4"]
    print(torch.cuda.is_available())
    x = torch.rand([1, 3, 227, 227]).cuda()
    for model_name in models_list:
        infer_eval_autograd(model_name, x)