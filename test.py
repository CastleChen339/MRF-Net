import os
import torch
from torchvision import transforms
from PIL import Image
import argparse
import math
from models import AODNet, C2PNet, dehazeformer, FFA, GridDehazeNet, MixDehazeNet, models, models_wb
from tqdm import tqdm


def split_image(image, block_size=(1000, 1000)):
    width, height = image.size
    blocks = []
    print("Spliting image...")
    for i in tqdm(range(0, width, block_size[0])):
        for j in range(0, height, block_size[1]):
            right = i if i + block_size[0] <= width else width - block_size[0]
            lower = j if j + block_size[1] <= height else height - block_size[1]
            block = image.crop((right, lower, right + block_size[0], lower + block_size[1]))
            blocks.append(block)
    return blocks


def join_blocks(blocks, original_size):
    block_size = blocks[0].size
    full_image = Image.new(blocks[0].mode, original_size)
    num_blocks_x = math.ceil(original_size[0] / block_size[0])
    num_blocks_y = math.ceil(original_size[1] / block_size[1])
    print("Joining results...")
    i = 0
    for x_index in tqdm(range(num_blocks_x)):
        for y_index in range(num_blocks_y):
            x_position = x_index * block_size[0]
            y_position = y_index * block_size[1]
            if (x_index + 1) * block_size[0] > original_size[0]:
                x_position = original_size[0] - block_size[0]
            if (y_index + 1) * block_size[1] > original_size[1]:
                y_position = original_size[1] - block_size[1]
            full_image.paste(blocks[i], (x_position, y_position))
            i += 1
    return full_image


def unnormalize(tensor, mean, std):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f'Using device: {device}')
    # model = models.IRDecloud().to(device)
    # model = models_wb.IRDecloudTest().to(device)
    # model = AODNet.dehaze_net().to(device)
    model = C2PNet.C2PNet(gps=3, blocks=19).to(device)
    # model = dehazeformer.dehazeformer_b().to(device)
    # model = FFA.FFA(gps=3,blocks=19).to(device)
    # model = GridDehazeNet.GridDehazeNet().to(device)
    # model = gunet.gunet_b().to(device)
    # model = MixDehazeNet.MixDehazeNet_b().to(device)
    print("Loading model: {}".format(args.model_path))
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(args.model_path))
    if args.test_img is None:
        raise FileExistsError("Please input path of test img.")
    cloud_img_list = []
    folder_flag = 0
    if not os.path.exists(args.test_img):
        raise FileExistsError("Please input path of test img.")
    if os.path.isfile(args.test_img):
        cloud_img_list.append(args.test_img)
    else:
        folder_flag = 1
        cloud_img_list = os.listdir(args.test_img)
    for path in cloud_img_list:
        if folder_flag == 0:
            cloud_img = Image.open(path).convert('L')
        else:
            cloud_img = Image.open(os.path.join(args.test_img, path)).convert('L')
        input_blocks = split_image(cloud_img, block_size=(args.block_size, args.block_size))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.34807074], [0.11415784])])
        res_blocks = []
        print("Cloud removing...")
        with torch.no_grad():
            for block in tqdm(input_blocks):
                input_tensor = transform(block).unsqueeze(0).to(device)
                print(input_tensor.shape)
                out = unnormalize(model(input_tensor), [0.34807074], [0.11415784])
                # out = unnormalize(input_tensor, [0.34807074], [0.11415784])
                block_image = transforms.ToPILImage()(out.clamp(0, 1).squeeze(0))
                res_blocks.append(block_image)
        res_img = join_blocks(res_blocks, cloud_img.size)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        print("Saving result...")
        res_img.save(os.path.join(args.save_dir, os.path.basename(path)))
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--test_img', type=str, default=r'./dataset/test/A1_small_cloud.png', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default=r'./checkpoints/self_batch_best_noloss2.pth', help='Path to pretrained model')
    parser.add_argument('--save_dir', type=str, default=r'./res', help='Path to save predict results')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA')
    parser.add_argument('--block_size', type=int, default=960, help='Block size for imput image')
    args = parser.parse_args()
    main(args)
