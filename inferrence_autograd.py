import os
import sys
from model import *
#from torch.profiler import profile, record_function, ProfilerActivity

# Directory
os.chdir(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_eval_autograd(model_name, path, inputData):
    print("Start 1000 Inferrence：", model_name+"@"+path)

    model = eval(model_name)(num_classes=5).to(device)
    # load model weights
    weights_path = "./weights/" + path
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    print("using {} device.".format(device))

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i in range(1000):
            model(inputData)
    print(prof)

if __name__ == '__main__':

    sys.argv.__delitem__(0);
    print('Inferrence Scheduled: ', str(sys.argv))

    inputData = torch.rand([1, 3, 227, 227]).cuda()

    for modelName in sys.argv:
        if "@" in modelName:
            modelAndPath = modelName.split("@")
            modelName = modelAndPath[0]
            path = modelAndPath[1]
            infer_eval_autograd(modelName, path, inputData)
        else:
            infer_eval_autograd(modelName, path, inputData)