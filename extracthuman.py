import cv2
import numpy as np
import os
import io
import argparse
import torch
import time
from torchvision import transforms
from PIL import Image
from rembg.bg import remove
from rembg.session_factory import new_session

def load_img(img_file):

    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not img_file.endswith("png"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img

def get_image_mask(img_file):

    mask_to_origin_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0, ), (1.0, ))
    ])
    img_ori = load_img(img_file)
    with torch.no_grad():
        buf = io.BytesIO()
        Image.fromarray(img_ori).save(buf, format='png')

        a=buf.getvalue()
        t1=time.time()
        b=remove(a, session=new_session("u2net"))
        t2=time.time()
        t=t2-t1
        print(t, 's')
        img_pil = Image.open(
            io.BytesIO(b)).convert("RGBA")

        # img_pil = Image.open(
        #     io.BytesIO(remove(buf.getvalue()))).convert("RGBA")
    img_mask = torch.tensor(1.0) - (mask_to_origin_tensor(img_pil.split()[-1]) <
                                    torch.tensor(0.5)).float()

    return img_mask

def apply_mask(image, mask):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0,
                                  255,
                                  image[:, :, c])
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default=r"Siltest/50ORG.jpg")
    parser.add_argument("--save_path", type=str, default=r"Siltest/")
    parser.add_argument("--save_seg", action="store_true")
    args = parser.parse_args()

    img = cv2.imread(args.img_path, cv2.IMREAD_UNCHANGED)
    base=os.path.basename(args.img_path)
    name=os.path.splitext(base)[0]
    mask=get_image_mask(args.img_path)

    if args.save_seg:
        cv2.imwrite(args.save_path+name+'_image.png',apply_mask(img,np.squeeze(mask.numpy(),0)))
    cv2.imwrite(args.save_path+name+'_mask.png',np.squeeze(mask.numpy(),0)*255)
    print(args.save_path+name+'_mask.png')
