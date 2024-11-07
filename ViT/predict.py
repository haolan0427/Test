import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224_in21k as create_model


def main():

    # 使用的是GPU还是CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 加载图像
    img_path = "./LoadImage/tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path) # 确保img_path存在
    img = Image.open(img_path)
    plt.imshow(img)

    # 对载入图像做处理
    img = data_transform(img)

    # 修改载入图像的shape
    img = torch.unsqueeze(img, dim=0)

    # 图片种类及其标签的对应文件
    json_path = './ClassJson/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型：在vit_model.py文件中
    model = create_model(num_classes=5, has_logits=False).to(device)


    # 加载已有的模型参数
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    # 预测
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()     # callable，调用forward函数
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
