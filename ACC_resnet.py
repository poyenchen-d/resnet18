import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
import tensorflow as tf
import cv2
from torch.utils.data import DataLoader, ConcatDataset, Subset , Dataset
from torchvision import datasets



char_classes = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
char_to_idx = {char: idx for idx, char in enumerate(char_classes)}
num_classes = len(char_classes)

# 1. 定義生成隨機字母、數字及其邊框標註的數據集


def load_emnist_dataset(transform):###手寫字符
    emnist_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    return emnist_dataset

def generate_yolo_label(image_size, class_id):
    img_w, img_h = image_size
    # x_center, y_center, width, height 相对于图片大小归一化
    x_center = 0.5
    y_center = 0.5
    width = 1.0
    height = 1.0
    return [class_id, x_center, y_center, width, height]
class EMNISTWithYOLO(torch.utils.data.Dataset):
    def __init__(self, emnist_dataset):
        self.emnist_dataset = emnist_dataset

    def __len__(self):
        return len(self.emnist_dataset)

    def __getitem__(self, idx):
        image, target = self.emnist_dataset[idx]  # 获取图像及其标签
        image_size = (image.shape[2], image.shape[1])  # (宽, 高)
        yolo_label = generate_yolo_label(image_size, target)  # 生成 YOLO 格式的标签
        return image, yolo_label





class MYDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=(128, 128)):
        """
        :param root_dir: 数据集根目录，包含图像和标签文件
        :param transform: 图像预处理操作
        :param image_size: 图像目标大小
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_size = image_size
        self.image_files = []  # 保存所有图像文件名
        self.annotations = {}   # 保存每个图像文件对应的标签
        
        # 加载类映射
        self.class_map = self._load_class_map()

        # 遍历数据文件夹，获取所有图像和对应的标签
        self._parse_data()

    def _load_class_map(self):
        """
        读取类映射文件，建立字符与类编号的映射
        """
        class_map_file = os.path.join(self.root_dir, "classes.txt")
        with open(class_map_file, 'r') as f:
            class_list = [line.strip() for line in f.readlines()]
        
        # 建立类编号到字符的映射
        class_map = {str(idx): char for idx, char in enumerate(class_list)}
        return class_map

    def _parse_data(self):
        """
        遍历文件夹，解析所有图像和标签文件
        """
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                image_path = os.path.join(self.root_dir, file_name)
                annotation_file = os.path.splitext(file_name)[0] + ".txt"
                annotation_path = os.path.join(self.root_dir, annotation_file)

                # 只处理有对应标签文件的图像
                if os.path.exists(annotation_path):
                    self.image_files.append(image_path)
                    # 解析标签文件
                    with open(annotation_path, 'r') as f:
                        annotations = [line.strip().split() for line in f.readlines()]
                    self.annotations[image_path] = annotations

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 获取图像路径
        img_path = self.image_files[index]

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        
        # 图像预处理：调整大小和转换为 tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # 确保在没有 transform 的情况下仍然转换为 tensor

        # 获取对应的标签
        labels = self.annotations[img_path]
        boxes = []  # 保存所有目标边界框
        texts = []  # 保存所有目标类别文本

        for label in labels:
            class_id, x_center, y_center, width, height = label
            # 转换为 float 类型
            x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
            
            # 归一化边界框坐标到 0-1 之间
            box = [x_center, y_center, width, height]  # 这里可以直接使用调整后的值
            boxes.append(box)

            # 获取类别字符
            texts.append(self.class_map[class_id])

        # 转换为 tensor 格式
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        # 返回 图像, 边界框 (tensor 格式), 文本标签
        return image, boxes_tensor, texts
    
    
def collate_fn(batch):
    images = []
    boxes = []
    texts = []

    max_boxes = 0  # 记录最大边框数量

    for item in batch:
        if len(item) == 3:  # 自定义数据集 MYDataset 返回 (image, boxes, texts)
            image, box, text = item
            images.append(image)
            boxes.append(box)
            texts.append(text)
            max_boxes = max(max_boxes, box.size(0))  # 更新最大边框数量
        elif len(item) == 2:  # EMNIST 返回 (image, label)
            image, _ = item
            images.append(image)
            boxes.append(torch.zeros((1, 4)))  # 对于 EMNIST 数据，生成一个假的 box
            texts.append("EMNIST")  # 对于 EMNIST 数据，使用一个占位符
            max_boxes = max(max_boxes, 1)  # 更新最大边框数量

    images_tensor = torch.stack(images, dim=0)
    
    # 填充 boxes
    padded_boxes = []
    for box in boxes:
        # 填充每个 box 以匹配最大边框数量
        if box.size(0) < max_boxes:
            padding = torch.zeros((max_boxes - box.size(0), 4))  # 生成填充
            padded_box = torch.cat((box, padding), dim=0)
        else:
            padded_box = box
        padded_boxes.append(padded_box)
    
    boxes_tensor = torch.stack(padded_boxes, dim=0)  # 现在 boxes_tensor 的形状一致
    return images_tensor, boxes_tensor, texts




        



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

# 定义ResNet层
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

# 定义ResNet18模型
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=62, input_channels=3):
        super(CustomResNet18, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 定义每一层的残差块
        self.layer1 = make_layer(BasicBlock, 64, 64, blocks=2)
        self.layer2 = make_layer(BasicBlock, 64, 128, blocks=2, stride=2)
        self.layer3 = make_layer(BasicBlock, 128, 256, blocks=2, stride=2)
        self.layer4 = make_layer(BasicBlock, 256, 512, blocks=2, stride=2)

        # 预测部分
        self.box_head = nn.Conv2d(512, 4, kernel_size=1)  # 预测边框
        self.label_head = nn.Conv2d(512, num_classes, kernel_size=1)  # 预测标签

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 边框和标签预测
        boxes = self.box_head(x)
        labels = self.label_head(x)

        # 展平并计算平均值
        boxes = torch.flatten(boxes, start_dim=2).mean(dim=-1)
        labels = torch.flatten(labels, start_dim=2).mean(dim=-1)

        return boxes, labels
            
    def _initialize_weights(self):
            # 权重初始化函数
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)



def train_model():
    user_input = input("Enter 1 to continue training from existing weights, 0 to start fresh: ").strip()
    load_existing_model = user_input == '1'

    
    batch_size = 4
    epochs = 100000

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),  # 将 1 通道扩展为 3 通道
    transforms.ToTensor()
    ])
    my_dataset = MYDataset(root_dir="d:\\NLP\\resnet18_V2\\train", transform=transform)

    # 加载 EMNIST 数据集
    def load_emnist_dataset(transform, num_samples=None):
        emnist_dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
        if num_samples:
            indices = random.sample(range(len(emnist_dataset)), num_samples)
            emnist_dataset = Subset(emnist_dataset, indices)
        return emnist_dataset

    # 选择 EMNIST 样本数量
    num_samples = 300000
    emnist_dataset = load_emnist_dataset(transform, num_samples=num_samples if num_samples > 0 else None)
    # 包装数据集
    emnist_with_yolo = EMNISTWithYOLO(emnist_dataset)
    # 组合 MYDataset 和 EMNIST 数据集
    combined_dataset = ConcatDataset([my_dataset, emnist_dataset])

    # 创建数据加载器
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 打印一下数据信息以验证
    print(f"Total number of samples: {len(combined_dataset)}") 


    #  cpu 
    # model = CustomResNet18(num_classes=num_classes)
    #  gpu
    model = CustomResNet18(num_classes=num_classes).to(device)
    if load_existing_model:
        model.load_state_dict(torch.load("model_weights.pth"))
        print("Loaded existing model weights.")
    else:
        print("Starting training from scratch.")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion_box = nn.SmoothL1Loss()
    i = 0
    for epoch in range(epochs):
        running_loss = 0.0
        
        i = i + 1
        for images, true_boxes, _ in tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]"):
            images = images.to(device)  #to gpu
            true_boxes = true_boxes.to(device)  #to gpu
            
            optimizer.zero_grad()
            predicted_boxes, _ = model(images)
            loss_box = criterion_box(predicted_boxes, true_boxes[:, 0, :4])
            loss_box.backward()
            optimizer.step()
            running_loss += loss_box.item()
            
            if(i >=2):
                model.cpu()  # 将模型从GPU移到CPU
                print("Training complete. Model weights saved to 'model_weights.pth'.")

                torch.save(model.state_dict(), "C:\\Users\\ian08\\model_weights.pth")
                model.to(device)  # 如果后续还需要继续在GPU上训练或推理，可以再移回GPU
                i = 0
            
        print(f"Epoch [{epoch + 1}/{epochs}], Box Loss: {running_loss / len(dataloader):.8f}")




def validate_model():
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = MYDataset(
        root_dir="d:\\NLP\\resnet18_V2\\train",
        transform=transform
    )

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


def capture_from_camera():
    model = CustomResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load("C:\\Users\\ian08\\model_weights.pth"))

    model.eval()
    
    if torch.cuda.is_available():
        model = model.to(device)

    # 使用 OpenCV 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    while True:
        # 读取摄像头的帧
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 将 BGR 图像转换为 RGB（PIL 图像使用 RGB）
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)

        # 转换图像为模型输入格式
        input_img = transform(img_pil).unsqueeze(0)
        if torch.cuda.is_available():
            input_img = input_img.to(device)

        # 模型进行预测
        with torch.no_grad():
            predicted_boxes, predicted_labels = model(input_img)
            predicted_box = predicted_boxes[0].cpu().numpy()
            predicted_label_idx = predicted_labels.argmax(dim=1).item()
            predicted_text = char_classes[predicted_label_idx]

        # 获取图像尺寸和预测框的位置
        height, width, _ = frame.shape
        x_center, y_center, box_width, box_height = predicted_box
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)

        # 绘制预测的边界框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {predicted_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # 显示图像
        cv2.imshow("Camera", frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows() 



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # train_model()
    capture_from_camera()
    #validate_model()
    predict_and_show("d:\\NLP\\resnet18_V2\\train\\3n3D.jpg")
