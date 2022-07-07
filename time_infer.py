import os
import torch
from model import *


# Directory
os.chdir(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def infer_eval_AlexNet(model_name):
    print("Start Eval：", model_name)

    model = eval(model_name)(num_classes=5).to(device)
    # load model weights
    weights_path = "./weights_val1/{}.pth".format(model_name)
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    print("using {} device.".format(device))

    x = torch.rand([1, 3, 227, 227]).cuda()

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        model(x)
    print(prof)

if __name__ == '__main__':
    models_list = ["AlexNet","AlexNet_without_conv1", "AlexNet_without_conv2", "AlexNet_without_conv3",
                   "AlexNet_without_conv4", "AlexNet_without_conv5", "AlexNet_without_BothFC"]
    print(torch.cuda.is_available())
    for model_name in models_list:
        infer_eval_AlexNet(model_name)