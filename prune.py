import torch
import torch.nn.utils.prune as prune
from torch.autograd import Variable
from cbam.imagenet import create_resnet
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
def unstructure_prune(model, pre):
    prune_number = 0
    all_number = 0
    for name, module in model.named_modules():
        print(name)
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pre)
            prune_number = prune_number + torch.sum(module.weight == 0)
        objlist = dir(module)
        if 'weight' in objlist:
            all_number = all_number + module.weight.nelement()
        # prune 40% of connections in all linear layers
        # elif isinstance(module, torch.nn.Linear):
        #     prune.l1_unstructured(module, name='weight', amount=0.4)
        #     prune_number = prune_number + torch.sum(module.weight == 0)
        #     all_number = all_number + module.weight.nelement()
    # print(100.*float(prune_number)/float(all_number))
    print('Pruned params: %.2fM \t Total params: %.2fM' % (prune_number/1000000.0, all_number/1000000.0))
    return model

def save_checkpoint(checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(filename)
    torch.save(checkpoint, filepath)

def test(model, loader, name):
    model.cuda()
    model.eval()

    correct = 0
    for data, target in loader:

        data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\n{}: Accuracy: {}/{} ({:.0f}%)\n'.format(name, correct,
          len(loader.dataset), 100. * correct / len(loader.dataset)))


data = 'skin'
depth = 50
arch = 'CBAM'
batch_size = 16
# prunerate = 0.2
# Data loading code
if data == 'diabetic':
    data_dir = '/workspace/cpfs-data/code/medical_image/diabeticV3'
    num_class = 5
elif data == 'skin':
    data_dir = '/workspace/cpfs-data/code/medical_image/HAM10000V2'
    num_class = 7
elif data == 'chest':
    data_dir = '/workspace/cpfs-data/code/medical_image/ChestXRayV3'
    num_class = 15

checkpoint = 'checkpoint/{}/{}_resnet{}.pth.tar'.format(data, data, (depth if arch is '' else str(depth) +'_' + arch.lower()))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    normalize])

image_datasets = datasets.ImageFolder(data_dir, data_transforms)
train_size = int(0.8 * len(image_datasets))
val_size = len(image_datasets) - train_size
train_set, val_set = torch.utils.data.random_split(image_datasets, [train_size, val_size])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, pin_memory=True)

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

model = create_resnet(depth, num_class, arch)
model = model.to(device)
checkpoint = torch.load(checkpoint)
model.load_state_dict(checkpoint['state_dict'])

model_name = f'{data}_resnet{depth}'
# test(model, val_loader, f'{model_name}')
model_0_2 = unstructure_prune(model, 0.2)
# checkpoint['state_dict'] = model_0_2.state_dict()
# save_checkpoint(checkpoint,f'checkpoint/{data}/{data}_resnet{depth}_prune.pth.tar')

# test(model_0_2, val_loader, f'{model_name} {0.2} Unstructural Pruning Model')

# model_0_4 = unstructure_prune(model, 0.4)
# test(model_0_4, val_loader, f'{model_name} {0.4} Unstructural Pruning Model')

# model_0_6 = unstructure_prune(model, 0.6)
# test(model_0_6, val_loader, f'{model_name} {0.6} Unstructural Pruning Model')

# model_0_8 = unstructure_prune(model, 0.8)
# test(model_0_8, val_loader, f'{model_name} {0.8} Unstructural Pruning Model')