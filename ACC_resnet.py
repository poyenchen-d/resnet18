import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 字符集映射
char_classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
char_to_idx = {char: idx for idx, char in enumerate(char_classes)}
num_classes = len(char_classes)

# 1. 定義生成隨機字母、數字及其邊框標註的數據集
class SynthTextDataset(Dataset):
    def __init__(self, num_samples=100000, transform=None, image_size=(128, 128)):
        self.num_samples = num_samples
        self.transform = transform
        self.image_size = image_size
        self.font = ImageFont.load_default()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = random.choice(char_classes)  # 隨機選擇字符
        img = Image.new('RGB', self.image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        text_size = draw.textbbox((0, 0), text, font=self.font)
        width, height = text_size[2] - text_size[0], text_size[3] - text_size[1]
        x_center = random.randint(width // 2, self.image_size[0] - width // 2)
        y_center = random.randint(height // 2, self.image_size[1] - height // 2)
        draw.text((x_center - width // 2, y_center - height // 2), text, font=self.font, fill=(0, 0, 0))

        label_idx = char_to_idx[text]
        box = [x_center / self.image_size[0], 
               y_center / self.image_size[1], 
               width / self.image_size[0], 
               height / self.image_size[1], 
               label_idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([box], dtype=torch.float32), text


# 2. 定義ResNet模型的基本塊
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# 定義ResNet層
def make_layer(block, in_channels, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample))
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels))

    return nn.Sequential(*layers)


# 3. 定義ResNet18模型
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=62):
        super(CustomResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   #輸入 3 通道，輸出 64 通道，7x7 的卷積核，步距為 2，填充為 3
        self.bn1 = nn.BatchNorm2d(64)   #batch normalization
        self.relu = nn.ReLU(inplace=True)   #ReLU
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     #最大池化層：3x3 的池化核，步距為 2，填充為 1

        self.layer1 = make_layer(BasicBlock, 64, 64, blocks=2) #resNet第1層 通道數為 64，不改變特徵圖大小
        self.layer2 = make_layer(BasicBlock, 64, 128, blocks=2, stride=2)   #resNet第2層 通道數為 128，特徵圖/2
        self.layer3 = make_layer(BasicBlock, 128, 256, blocks=2, stride=2)  #resNet第3層 通道數為 256，特徵圖/2
        self.layer4 = make_layer(BasicBlock, 256, 512, blocks=2, stride=2)  #resNet第4層 通道數為 512，特徵圖/2

        self.box_head = nn.Conv2d(512, 4, kernel_size=1)  #預測邊框x_center, y_center, width, height
        self.label_head = nn.Conv2d(512, num_classes, kernel_size=1)    # 預測標籤的卷積層

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        boxes = self.box_head(x)
        labels = self.label_head(x)

        boxes = torch.flatten(boxes, start_dim=2).mean(dim=-1)
        labels = torch.flatten(labels, start_dim=2).mean(dim=-1)

        return boxes, labels


def train_model():
    user_input = input("Enter 1 to continue training from existing weights, 0 to start fresh: ").strip()
    load_existing_model = user_input == '1'

    sample_size = 50000
    batch_size = 32
    epochs = 100

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = SynthTextDataset(num_samples=sample_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CustomResNet18(num_classes=num_classes)
    if load_existing_model:
        model.load_state_dict(torch.load("model_weights.pth"))
        print("Loaded existing model weights.")
    else:
        print("Starting training from scratch.")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion_box = nn.SmoothL1Loss()

    for epoch in range(epochs):
        running_loss = 0.0
        for images, true_boxes, _ in tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]"):
            optimizer.zero_grad()
            predicted_boxes, _ = model(images)
            loss_box = criterion_box(predicted_boxes, true_boxes[:, 0, :4])
            loss_box.backward()
            optimizer.step()
            running_loss += loss_box.item()
        torch.save(model.state_dict(), "model_weights.pth")
        print(f"Epoch [{epoch + 1}/{epochs}], Box Loss: {running_loss / len(dataloader):.8f}")

    print("Training complete. Model weights saved to 'model_weights.pth'.")



def validate_model():
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = SynthTextDataset(num_samples=1000, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = CustomResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()

    correct_count = 0
    total_count = 0

    for sample_img, true_boxes, text in tqdm(dataloader):
        predicted_boxes, _ = model(sample_img)
        _, _, height, width = sample_img.shape

        true_box = true_boxes[0].detach().numpy()
        pred_box = predicted_boxes[0].detach().numpy()

        true_x1 = (true_box[0, 0] - true_box[0, 2] / 2) * width
        true_y1 = (true_box[0, 1] - true_box[0, 3] / 2) * height
        true_x2 = (true_box[0, 0] + true_box[0, 2] / 2) * width
        true_y2 = (true_box[0, 1] + true_box[0, 3] / 2) * height

        pred_x1 = (pred_box[0] - pred_box[2] / 2) * width
        pred_y1 = (pred_box[1] - pred_box[3] / 2) * height
        pred_x2 = (pred_box[0] + pred_box[2] / 2) * width
        pred_y2 = (pred_box[1] + pred_box[3] / 2) * height

        if abs(true_x1 - pred_x1) <= 1.5 and abs(true_y1 - pred_y1) <= 1.5:
            correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count
    print(f"Model accuracy on validation set: {accuracy:.2%}")

# 6. 定義可視化函數，用於預測新圖片並顯示結果
def predict_and_show(image_path):
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    
    # 加载图片
    img = Image.open(image_path).convert('RGB')
    input_img = transform(img).unsqueeze(0)
    
    # 加载模型并进行预测
    model = CustomResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval()
    
    with torch.no_grad():
        predicted_boxes, predicted_labels = model(input_img)
        predicted_box = predicted_boxes[0].numpy()
        predicted_label_idx = predicted_labels.argmax(dim=1).item()
        predicted_text = char_classes[predicted_label_idx]

    # 获取图像尺寸和预测框的位置
    width, height = img.size
    x_center, y_center, box_width, box_height = predicted_box
    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)

    # 绘制预测结果
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.text(x1, y1 - 10, f"Predicted: {predicted_text}", color='red', fontsize=12, backgroundcolor='white')
    plt.show()



if __name__ == "__main__":
    train_model()
    validate_model()
    predict_and_show("d:\\NLP\\resnet18_V1\\test2.jpg")
