# !conda install --yes --prefix {sys.prefix} scikit-image
# !pip3 install scikit-image

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
parser.add_argument('--model', type=str, default='default', metavar='M',
					help="""Options are default, P2Q7DefaultChannelsNet,
					P2Q7HalfChannelsNet, P2Q7DoubleChannelsNet,
					P2Q8BatchNormNet, P2Q9DropoutNet, P2Q10DropoutBatchnormNet,
					P2Q11ExtraConvNet, P2Q12RemoveLayerNet, and P2Q13UltimateNet.""")
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
	def __init__(self, df):
		self.df = df 

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		photo_name = self.df.iloc[idx].name
		file_name = get_photo_file(photo_name)
		img = io.imread(file_name)
		img = resize_img(img) 
		label = self.df.iloc[idx].label
		return (torch.Tensor(img), torch.Tensor([label]))

biz_df = pd.read_csv(args.data_dir + "clean_business.csv").set_index("business_id")
photo_df = pd.read_csv(args.data_dir + "clean_photo.csv").set_index("photo_id")
df = photo_df.copy(deep=True)
df["label"] = pd.Series(biz_df.loc[df["business_id"]]["stars"]).tolist()
df = df.sample(frac=1)

# train_df = df.iloc[0:int(len(df) * 0.7)]
# val_df = df.iloc[int(len(df) * 0.7):]

val_df = df.iloc[0:3000]
train_df = df.iloc[3000:10000]


train_dataset = YelpDataset(train_df)
val_dataset = YelpDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=2)

val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),
                        shuffle=True, num_workers=2)


# val_img, val_label = val_loader[0]
# val_img, val_label = Variable(val_img), Variable(val_label)


class BasicNet(nn.Module):
	def __init__(self, dropout_rate=0.5):
		super(BasicNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 5, kernel_size=5)
		self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
		self.fc1 = nn.Linear(28090, 100)
		self.fc2 = nn.Linear(100, 1)
		self.dropout_rate = dropout_rate

	def forward(self, x):
		x = x.permute(0, 3, 1, 2)
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(-1, 28090)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=self.dropout_rate, training=self.training)
		x = self.fc2(x)
		return x

cuda_is_avail = torch.cuda.is_available()

model = BasicNet()
if cuda_is_avail:
	model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


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
		# print("Training step " + str(i) + ": " + str(loss.data.item()))
		i += 1
		loss_list.append(loss)

	total_loss = 0
	for loss in loss_list:
		total_loss += loss
	total_loss /= i

	return total_loss, loss_list

def test():

	model.eval()

	for input_batch, label_batch in val_loader:
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

	print("Val loss: " +  str(loss.data.item()))
	return loss

for module in model.children():
	module.reset_parameters()

train_loss, train_loss_list = train_epoch()
val_loss = test()

