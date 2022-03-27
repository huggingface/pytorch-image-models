import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
	def __init__(self):
		self.imgs_path = "Dog_Cat_Dataset/"
		file_list = glob.glob(self.imgs_path + "*")
		print(file_list)
		self.data = []
		for class_path in file_list:
			class_name = class_path.split("/")[-1]
			for img_path in glob.glob(class_path + "/*.jpeg"):
				self.data.append([img_path, class_name])
		print(self.data)
		self.class_map = {"dogs" : 0, "cats": 1}
		self.img_dim = (416, 416)
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path, class_name = self.data[idx]
		img = cv2.imread(img_path)
		img = cv2.resize(img, self.img_dim)
		class_id = self.class_map[class_name]
		img_tensor = torch.from_numpy(img)
		img_tensor = img_tensor.permute(2, 0, 1)
		class_id = torch.tensor([class_id])
		return img_tensor, class_id

if __name__ == "__main__":
	dataset = CustomDataset()		
	data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
	for imgs, labels in data_loader:
		print("Batch of images has shape: ",imgs.shape)
		print("Batch of labels has shape: ", labels.shape)