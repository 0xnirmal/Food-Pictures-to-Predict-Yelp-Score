import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data
from skimage import io, transform
from skimage.measure import block_reduce
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models, transforms
import argparse


parser = argparse.ArgumentParser(description='Deep Learning JHU Assignment 1 - Fashion-MNIST')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
					help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='TB',
					help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
					help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
					help='learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='sgd', metavar='O',
					help='Optimizer options are sgd, p1sgd, adam, rms_prop')
parser.add_argument('--momentum', type=float, default=0.5, metavar='MO',
					help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='I',
					help="""how many batches to wait before logging detailed
							training status, 0 means never log """)
parser.add_argument('--dataset', type=str, default='mnist', metavar='D',
					help='Options are mnist and fashion_mnist')
parser.add_argument('--data_dir', type=str, default='data/', metavar='F',
					help='Where to put data')
parser.add_argument('--photos_dir', type=str, default='../HDD/photos/', metavar='F',
					help='Where to put photos')
parser.add_argument('--name', type=str, default='', metavar='N',
					help="""A name for this training run, this
							affects the directory so use underscores and not spaces.""")
parser.add_argument('--model', type=str, default='basic', metavar='M',
					help="""Options are basic, inter""")
parser.add_argument('--print_log', action='store_true', default=False,
					help='prints the csv log when training is complete')
parser.add_argument('--dropout_rate', type=float, default=0.5)
parser.add_argument("-f", "--jupyter-json")
parser.add_argument('--loss', type=str, default='l1', metavar='LOSS',
					help='l1, mse')

required = object()
args = parser.parse_args()

def get_photo_file(photo_name):
	return args.photos_dir + photo_name + ".jpg"

def resize_img(img):
	return transform.resize(img, (224, 224))

class YelpDataset(torch.utils.data.Dataset):
	def __init__(self, df, transform=None):
		self.df = df 
		self.transform = transform

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		photo_name = self.df.iloc[idx].name
		file_name = get_photo_file(photo_name)
		img = io.imread(file_name)
		img = torch.Tensor(resize_img(img)).permute(2, 0, 1) 
		label = torch.Tensor([self.df.iloc[idx].label])
		if self.transform is not None:
			img = self.transform(img)
		return (img, label)


df = pd.read_csv(args.data_dir + "clean_data.csv").set_index("photo_id")
df = df.sample(frac=1)

train_df = df.iloc[0:int(len(df) * 0.7)]
val_df = df.iloc[int(len(df) * 0.7):]

normalize = None
if args.model == "pretrained":
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
		std=[0.229, 0.224, 0.225])

train_dataset = YelpDataset(train_df, transform=normalize)
val_dataset = YelpDataset(val_df, transform=normalize)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
						shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, num_workers=2)

class AutoEncoder(nn.Module):
	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
			nn.ReLU(True),
			nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
			nn.ReLU(True),
		)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1),  # b, 16, 5, 5
			nn.ReLU(True),
			nn.ConvTranspose2d(16, 3, 3, stride=3, padding=1, output_padding=1),  # b, 8, 15, 15
			nn.Tanh()
		)

	def forward(self, x):
		x = self.encoder(x)
		print(x.shape)
		x = self.decoder(x)
		return x

def train_autoencoder():

	autoencoder = AutoEncoder()
	if cuda_is_avail:
		autoencoder.cuda()

	auto_optimizer = torch.optim.SGD(autoencoder.parameters(), lr=0.001)

	autoencoder.train()

	for epoch in range(10):
		i = 0
		for input_batch, _ in train_loader:
		
			input_batch = Variable(input_batch)
			if cuda_is_avail:
				input_batch = input_batch.cuda()

			output_batch = autoencoder(input_batch)

			loss = F.mse_loss(output_batch, input_batch)

			auto_optimizer.zero_grad()
			loss.backward()
			auto_optimizer.step()
			print(str(i) + "," + str(loss.data.item()))
			i += 1

			return autoencoder.encoder

class BasicNet(nn.Module):
	def __init__(self, dropout_rate=0.5):
		super(BasicNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
		self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
		self.fc1 = nn.Linear(28090, 100)
		self.fc2 = nn.Linear(100, 1)
		self.dropout_rate = dropout_rate

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 28090)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.fc2(x)
		return x

class IntermediateNet(nn.Module):
	def __init__(self):
		super(IntermediateNet, self).__init__()
		self.bn1 = nn.BatchNorm2d(3)
		self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
		self.bn2 = nn.BatchNorm2d(10)
		self.conv2 = nn.Conv2d(10, 10, kernel_size=6)
		self.bn3 = nn.BatchNorm2d(10)
		self.fc1 = nn.Linear(116640, 1024)
		self.fc2 = nn.Linear(1024, 1)

	def forward(self, x):
		x = self.bn1(x)
		x = self.bn2(F.relu(self.conv1(x)))
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.bn3(F.relu(F.max_pool2d(self.conv2(x),2)))
		x = F.dropout(x, p=0.3, training=self.training)
		x = x.view(-1, 116640)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

class PreTrainedNet(nn.Module):
	def __init__(self, orig_resnet):
		super().__init__()
		self.orig_resnet = orig_resnet
		self.final_linear = torch.nn.Linear(512, 1)

	def forward(self, x):
		x = self.orig_resnet.conv1(x)
		x = self.orig_resnet.bn1(x)
		x = self.orig_resnet.relu(x)
		x = self.orig_resnet.maxpool(x)

		x = self.orig_resnet.layer1(x)
		x = self.orig_resnet.layer2(x)
		x = self.orig_resnet.layer3(x)
		x = self.orig_resnet.layer4(x)

		x = self.orig_resnet.avgpool(x)
		x = x.view(x.size(0), -1)

		x = self.final_linear(x)
		return x

class AutoBasicNet(nn.Module):
	def __init__(self, encoder):
		super(AutoBasicNet, self).__init__()
		self.encoder = encoder
		self.conv1 = nn.Conv2d(8, 30, kernel_size=2)
		self.conv2 = nn.Conv2d(30, 10, kernel_size=3)
		self.fc1 = nn.Linear(28090, 100)
		self.fc2 = nn.Linear(100, 1)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		print(x.shape)
		x = x.view(-1, 28090)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

cuda_is_avail = torch.cuda.is_available()

if args.model == "basic":
	model = BasicNet()
elif args.model == "inter":
	model = IntermediateNet()
elif args.model == "pretrained":
	m = models.resnet18(pretrained=True)
	model = PreTrainedNet(m)
elif args.model == "autoencoder":
	print("Getting into autoencoder function")
	encoder = train_autoencoder()
	print("Getting into autoencoder function")
	model = AutoBasicNet(encoder)

if cuda_is_avail:
	if args.model == "pretrained":
		m.cuda()
	model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

if args.model == "pretrained":
	optimizer = torch.optim.SGD(model.final_linear.parameters(), lr=0.001)


def train_epoch():

	model.train()
	i = 0
	loss_list = []
	for input_batch, label_batch in train_loader:
	
		input_batch, label_batch = Variable(input_batch), Variable(label_batch)
		if cuda_is_avail:
			input_batch, label_batch = input_batch.cuda(), label_batch.cuda()
		output_batch = model(input_batch)

		if args.loss == "l1":
			loss = F.l1_loss(output_batch.squeeze(), label_batch.squeeze())
		elif args.loss == "mse":
			loss = F.mse_loss(output_batch.squeeze(), label_batch.squeeze())
		else:
			print("Invalid loss function")
			sys.exit(-1)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(str(i) + "," + str(loss.data.item()))
		i += 1
		loss_list.append(loss.data.item())

	total_loss = 0
	for loss in loss_list:
		total_loss += loss
	total_loss /= i

	return total_loss, loss_list

def test():
	model.eval()
	i = 0
	loss_list = []
	for input_batch, label_batch in val_loader:
	
		input_batch, label_batch = Variable(input_batch, volatile=True), Variable(label_batch, volatile=True)
		if cuda_is_avail:
			input_batch, label_batch = input_batch.cuda(), label_batch.cuda()
		output_batch = model(input_batch)

		if args.loss == "l1":
			loss = F.l1_loss(output_batch.squeeze(), label_batch.squeeze())
		elif args.loss == "mse":
			loss = F.mse_loss(output_batch.squeeze(), label_batch.squeeze())
		else:
			print("Invalid loss function")
			sys.exit(-1)

		print(str(i) + "," + str(loss.data.item()))
		i += 1
		loss_list.append(loss.data.item())
		del loss, input_batch, label_batch, output_batch

	total_loss = 0
	for loss in loss_list:
		total_loss += loss
	total_loss /= i

	return total_loss, loss_list


if args.model != "pretrained":
	for module in model.children():
		module.reset_parameters()

train_loss_list = []
val_loss_list = []

for i in range(args.epochs):
	print("Epoch " + str(i))
	print("Training:")
	train_loss, temp_train_loss_list = train_epoch()
	train_loss_list.extend(temp_train_loss_list)
	print("Validating...")
	val_loss, _ = test()
	val_loss_list.append(val_loss)

print("Training:")
for idx, val in enumerate(train_loss_list):
	print(str(idx) + "," + str(val))

print("Validation:")
for idx, val in enumerate(val_loss_list):
	print(str(idx) + "," + str(val))

