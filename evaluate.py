import os
import sys
import json
import torch
from torchvision import transforms, datasets, utils
from tqdm import tqdm
from model import *
from torch.utils.tensorboard import SummaryWriter

# Directory
os.chdir(os.getcwd())
image_path = os.path.abspath(os.path.join(os.getcwd(), "./dataset"))  # get data root path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(227),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((227, 227)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test": transforms.Compose([transforms.Resize((227, 227)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


def evaluate_AlexNet(model_name, path):
    print("Start Eval：", model_name)
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    # load model
    model = eval(model_name)(num_classes=5).to(device)
    # load model weights
    weights_path = "./weights/" + path
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    print("using {} device.".format(device))

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=1)
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        val_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in val_bar:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

    test_accurate = acc / test_num
    print('{} test accuracy: {}'.format(model_name,test_accurate))


if __name__ == '__main__':
    sys.argv.__delitem__(0);

    cmd_list = sys.argv;
    if sys.argv[0] == "-f" and len(sys.argv) > 1:
        cmd_list = open(sys.argv[1], "r").read().splitlines()

    print('Evaluating Scheduled: ', str(cmd_list))

    for modelName in cmd_list:
        if "@" in modelName:
            modelAndPath = modelName.split("@")
            modelName = modelAndPath[0]
            path = modelAndPath[1]
            evaluate_AlexNet(modelName, path)
        else:
            evaluate_AlexNet(modelName, modelName + ".pth")
