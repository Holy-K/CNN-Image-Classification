'''
CNN_Image_Classification
Kazuki Hori
Waseda University
Last update 2022/11/26
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import cv2
import keyboard


#GPUの設定
print('torch.cuda.is_available:',torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#画像の前処理を定義
data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop(224),  # ランダムにトリミングして (224, 224)の形状にしてる
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),  # 50%の確率で水平方向に反転させる
        transforms.ToTensor(),  # Tensorに変換
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
    ]),
    'val': transforms.Compose([
        #transforms.Resize(256),  # 画像のサイズを(256, 256)にする
        #transforms.CenterCrop(224),  # (224, 224)にするために、サイズ変更された画像を中央で切り取る
        transforms.Resize(224),
        transforms.ToTensor(),  # Tensorに変換
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 平均値と標準偏差を指定して、結果のTensorを正規化
    ]),
}

#正規化をしない前処理
to_tensor_transforms = transforms.Compose([
        #transforms.RandomResizedCrop(224),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

#前処理の確認
print( 'data_transforms["train"]:', data_transforms["train"])
print( 'data_transforms["val"]:', data_transforms["val"])
print( 'to_tensor_transforms:', to_tensor_transforms)


#カスタムデータセットの定義
root = os.getcwd()
class CustomDataset(torch.utils.data.Dataset):  
    classes = ['A', 'B']

    def __init__(self, root, transform=None, train=True):
        # 指定する場合は前処理クラスを受け取る
        self.transform = transform
        # 画像とラベルの一覧を保持するリスト
        self.images = []
        self.labels = []
        # ルートフォルダーパス
        # 訓練の場合と検証の場合でフォルダわけ
        # 画像を読み込むファイルパスを取得
        if train == True:
            root_A_path = os.path.join(root, 'train', 'train_A')
            root_B_path = os.path.join(root, 'train', 'train_B')
        else:
            root_A_path = os.path.join(root, 'val', 'val_A')
            root_B_path = os.path.join(root, 'val', 'val_B')
        # Aの画像一覧を取得
        A_images = os.listdir(root_A_path)
        # ここではAをラベル０に指定
        A_labels = [0] * len(A_images)
        # Bの画像一覧を取得
        B_images = os.listdir(root_B_path)
        # ここではBをラベル１に指定
        B_labels = [1] * len(B_images)
        # 1個のリストにする
        for image, label in zip(A_images, A_labels):
            self.images.append(os.path.join(root_A_path, image))
            self.labels.append(label)
        for image, label in zip(B_images, B_labels):
            self.images.append(os.path.join(root_B_path, image))
            self.labels.append(label)
        
    def __getitem__(self, index):
        # インデックスを元に画像のファイルパスとラベルを取得
        image = self.images[index]
        label = self.labels[index]
        # 画像ファイルパスから画像を読み込む
        with open(image, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        # 前処理がある場合は前処理をいれる
        if self.transform is not None:
            image = self.transform(image)
        # 画像とラベルのペアを返却
        return image, label
        
    def __len__(self):
        # ここにはデータ数を指定
        return len(self.images)


# 訓練/テストデータのプロット
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')


while True:
    testProt_YorN=input('Do you need to test plot of your images? [Yes=0,No=1]\n')
    if str.isdigit(testProt_YorN) and int(testProt_YorN) == 0 :
        #訓練データのプロット
        custom_dataset = CustomDataset(root, to_tensor_transforms, train=True)
        custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                    batch_size=5, 
                                                    shuffle=True)
        for i, (images, labels) in enumerate(custom_loader):
            print('These are test plots of train images:')
            print('Labels of showing pictures are',labels.numpy())
            print('Close the window of Figure 1 if there are no problem.\n')
            show(torchvision.utils.make_grid(images, padding=1))
            plt.axis('off')
            plt.show()
            break

        # テストデータのプロット
        custom_dataset = CustomDataset(root, to_tensor_transforms,train=False)
        custom_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                                    batch_size=5, 
                                                    shuffle=True)
        for i, (images, labels) in enumerate(custom_loader):
            print('These are test plots of val images:')
            print('Labels of showing pictures are',labels.numpy())
            print('Close the window of Figure 1 if there are no problem.\n')
            show(torchvision.utils.make_grid(images, padding=1))
            plt.axis('off')
            plt.show()
            break
        break
    elif str.isdigit(testProt_YorN) and int(testProt_YorN) == 1 :
        break
    else:
        print('Please enter the correct key.')


#定義したDatasetとDataLoaderを使用
custom_train_dataset = CustomDataset(root, data_transforms["train"], train=True)
train_loader = torch.utils.data.DataLoader(dataset=custom_train_dataset,
                                           batch_size=5, 
                                           shuffle=True)
custom_test_dataset = CustomDataset(root, data_transforms["val"],train=False)
test_loader = torch.utils.data.DataLoader(dataset=custom_test_dataset,
                                           batch_size=5, 
                                           shuffle=False)
for i, (images, labels) in enumerate(train_loader):
    print('images.size():',images.size())
    print('images[0].size():',images[0].size())    
    print('labels[0].item():',labels[0].item())
    break

#ネットワークの定義
#全結合の次元を計算
size_check = torch.FloatTensor(10, 3, 224, 224)  # バッチサイズ: 10, チャンネル数: 3, 横幅: 224, 縦幅: 224
features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
#バッチサイズ10, 6×6のフィルターが256枚
#10バッチは残して、6×6×256を１次元に落とす=>6×6×256=9216
print('features(size_check).size():',features(size_check).size())
#バッチ１０の値を軸にして残りの次元を１次元へ落とした場合のTensorの形状をチェックすると9216。
print('features(size_check).view(size_check.size(0), -1).size():',features(size_check).view(size_check.size(0), -1).size())
#fc_sizeを全結合の形状として保持しておく
fc_size = features(size_check).view(size_check.size(0), -1).size()[1]
print('fc_size:',fc_size,'\n')

#アーキテクチャの定義
num_classes = 2

class AlexNet(nn.Module):
    #fc_sizeを引き渡す
    def __init__(self, num_classes, fc_size):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),#調整された線形単位関数を要素ごとに適用
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #fc_sizeで計算した形状を指定
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fc_size, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AlexNet(num_classes, fc_size).to(device)
criterion = nn.CrossEntropyLoss()
#最適化関数を設定
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)#lr:学習率。 学習率を大きくしすぎると発散し、小さくしすぎると収束まで遅くなる。
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.99), eps=1e-09)
#optimizer = optim.RMSprop(net.parameters(),lr=1e-4)
#optimizer = optim.SGD(net.parameters(),lr=1e-4)
print(net)


#学習
while True:
    num_epochs = input('Enter the nonnegative integer of the epochs.(The higher the number, the longer it takes.)\n')
    if str.isdigit(num_epochs) and int(num_epochs) >= 0 :
        num_epochs = int(num_epochs)
        break
    else:
       print('Please enter the correct nonnegative integer.')
print('Now learning...')
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
lastPrediction_labels=[] 
image_number=0
for epoch in range(num_epochs):
  #変数の初期化
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    
    #train
    net.train()
    for i, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()#勾配を初期化
      outputs = net(images)
      loss = criterion(outputs, labels)
      train_loss += loss.item()
      train_acc += (outputs.max(1)[1] == labels).sum().item()
      loss.backward()
      optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)
    
    #val
    net.eval()#推論モードに切り替え
    with torch.no_grad():
      for images, labels in test_loader:
        images = images.to(device)#.to(device):device（今回はcuda(GPU)）専用変数に設定
        labels = labels.to(device)
        outputs = net(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += (outputs.max(1)[1] == labels).sum().item()#item():テンソル(torch.Tensor)の要素をintやfloatとして取得
        if epoch == num_epochs-1:
          lastPrediction_labels.append([image_number * 5,(outputs.max(1)[1] == labels)])
          image_number+=1
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}' 
                   .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, train_acc=avg_train_acc,val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)
print('lastPrediction:',lastPrediction_labels)


#学習結果の可視化
import matplotlib.pyplot as plt
print('Close graphs if you want to continue.')
plt.figure()
plt.plot(range(num_epochs), train_loss_list, color='blue', linestyle='-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, color='green', linestyle='--', label='val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
plt.figure()
plt.plot(range(num_epochs), train_acc_list, color='blue', linestyle='-', label='train_acc')
plt.plot(range(num_epochs), val_acc_list, color='green', linestyle='--', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation accuracy')
plt.grid()
plt.show()


#TorchScriptModelの保存
while True:
    save_TorchScriptModel_YorN=input('Do you need to save TorchScriptModel? [Yes=0,No=1]\n')
    if str.isdigit( save_TorchScriptModel_YorN) and int( save_TorchScriptModel_YorN) == 0 :
        import datetime
        from tkinter import filedialog
        model_scripted = torch.jit.script(net) 
        now = datetime.datetime.now()
        model_scripted.save('Saved-TorchScriptModel_'+ now.strftime('%Y%m%d_%H%M%S') +'.pth') 
        print('Model is saved at',os.getcwd())
        break
    elif str.isdigit( save_TorchScriptModel_YorN) and int( save_TorchScriptModel_YorN) == 1 :
        break
    else:
      print('Please enter the correct key.')