
import os
import sys
import json

from torch.nn import Parameter
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm
from model import *
from torch.utils.tensorboard import SummaryWriter

# Global Parameters

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parameters
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(227),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((227, 227)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


def load_freeze_part_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        print("Copying " + name + " Data...")

        if (param.size() == own_state[name].size()):
            own_state[name].copy_(param)
        else:
            wShape = own_state[name].shape;
            if wShape[0] <= param.shape[0] and wShape[1] <= param.shape[1]:
                print(" pretrained " + name + ": ", param.size())
                print(" Current Model " + name + ": ", own_state[name].size()),
                sliced = param[0:wShape[0],0:wShape[1]]
                print(" Sliced to: ", sliced.size())
                own_state[name].copy_(sliced)

    #To only train the designated Layer. Freezing the rest of layers
    for name, param in model.named_parameters():
        Tolock = True;
        for key in model.layerToTrain:
            if key in name:
                Tolock = False;
                break;
        if Tolock:
            print("Freeze: " + name)
            param.requires_grad = False
        else:
            print("Unfreeze: " + name)


def train_AlexNet(model_name, ptmodel_path="", image_path = "./dataset", epochs = 100, batch_size = 32):

    # Directory
    image_path = os.path.abspath(os.path.join(os.getcwd(), image_path))  # get data root path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    starting = "scratch"
    if ptmodel_path == "":
        save_path = "./weights/" + model_name + ".pth" # Save Weights
    else:
        save_path = "./weights/" + model_name + "_from_" + ptmodel_path # Save Weights with Transfer Learning
        starting = ptmodel_path

    print("Start Training：", model_name, " from ", starting)
    net = eval(model_name)(num_classes=5)
    print("using {} device.".format(device))

    tb_writer = SummaryWriter("runs/{}".format(model_name))
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('../class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net.to(device)

    if ptmodel_path != "":
        pretrained_weights_path = "./weights/" + ptmodel_path
        assert os.path.exists(pretrained_weights_path), "file: '{}' dose not exist.".format(pretrained_weights_path)

        load_freeze_part_state_dict(net,torch.load(pretrained_weights_path))

    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        tb_writer.add_scalar("train_loss", running_loss / train_steps, epoch + 1)
        tb_writer.add_scalar("val_accuracy", val_accurate, epoch + 1)
    print('Finished Training')

if __name__ == '__main__':
    sys.argv.__delitem__(0);

    cmd_list = sys.argv;
    if sys.argv[0] == "-f" and len(sys.argv) > 1:
        cmd_list = open(sys.argv[1], "r").read().splitlines()

    print('Training Scheduled: ', str(cmd_list))

    for modelName in cmd_list:
        if "@" in modelName:
            modelAndPath = modelName.split("@")
            modelName = modelAndPath[0]
            path = modelAndPath[1]
            train_AlexNet(modelName, path)
        else:
            train_AlexNet(modelName)


