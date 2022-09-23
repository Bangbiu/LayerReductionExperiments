import os
import sys
import torch
from model import *
import torchprof
#from torch.profiler import profile, record_function, ProfilerActivity

# Directory
os.chdir(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_eval_torchprof(model_name, path, inputData):
    print("Start Inferrence for 1000 times：", model_name + "@" + path)

    model = eval(model_name)(num_classes=5).to(device)
    # load model weights
    weights_path = "./weights/" + path
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

    sys.argv.__delitem__(0);
    print('Inferrence Scheduled: ', str(sys.argv))

    inputData = torch.rand([1, 3, 227, 227]).cuda()

    for modelName in sys.argv:
        if "@" in modelName:
            modelAndPath = modelName.split("@")
            modelName = modelAndPath[0]
            path = modelAndPath[1]
            infer_eval_torchprof(modelName, path, inputData)
        else:
            infer_eval_torchprof(modelName, path, inputData)