import os
from PIL import Image
import base64
from io import BytesIO
import json
from tqdm import tqdm

index_file = "/mnt/v-yukangyang_blob/data/LAION-Glyph-1M-split/index.tsv"
laion_root = "/mnt/v-yukangyang_blob/data/LAION-Glyph-1M-split"
llava_caption_root = "/mnt/v-yukangyang_blob/data/LAION-Glyph-1M-llava-split/"
llava_llama2_caption_root = "/mnt/v-yukangyang_blob/data/LAION-Glyph-1M-llava-llama2-split/"
org_img_output_dir = "org_img"
condition_img_output_dir = "condition_img"
num_samples = 100

def main():
    with open(index_file, "r") as f:
        lines = f.readlines()

    cnt = 0
    with open("README.md", "w") as fout:
        head = "| Original Image | Condition Image | Original caption | Blip caption | LLAVA caption | LLAVA-LLAMA2 caption |\n | --- | --- | --- | --- | --- | --- |\n"
        fout.write(head)
        for line in tqdm(lines):
            img_id = line.strip()
            path_prefix = img_id.replace("\t", "/")
            data_file = os.path.join(laion_root, img_id.replace("\t", "/") + ".json")
            if not os.path.exists(data_file):
                continue
            llava_caption_path = os.path.join(llava_caption_root, path_prefix + ".txt")
            if not os.path.exists(llava_caption_path):
                continue
            llava_llama2_caption_path = os.path.join(llava_llama2_caption_root, path_prefix + ".txt")
            if not os.path.exists(llava_llama2_caption_path):
                continue

            with open(data_file, "r") as f:
                data = json.load(f)
            ori_img = Image.open(BytesIO(base64.b64decode(data["img_code"])))
            ori_img_path = os.path.join(org_img_output_dir, path_prefix + ".png")
            os.makedirs(os.path.dirname(ori_img_path), exist_ok=True)
            ori_img.save(ori_img_path)

            condition_img = Image.open(BytesIO(base64.b64decode(data["hint"])))
            condition_img_path = os.path.join(condition_img_output_dir, path_prefix + ".png")
            os.makedirs(os.path.dirname(condition_img_path), exist_ok=True)
            condition_img.save(condition_img_path)
            os.makedirs(os.path.dirname(condition_img_path), exist_ok=True)

            ori_caption = data["caption_origin"]
            blip_caption = data["caption_blip"]
            with open(llava_caption_path, "r") as f:
                llava_caption = f.read().strip().replace("\n", " ")
            with open(llava_llama2_caption_path, "r") as f:
                llava_llama2_caption = f.read().strip().replace("\n", " ")
            
            content = "| ![Original Image](./" + ori_img_path + ") | ![Condition Image](./" + condition_img_path + ") | " + ori_caption + " | " + blip_caption + " | " + llava_caption + " | " + llava_llama2_caption + " |\n"
            fout.write(content)

            cnt += 1
            if cnt >= num_samples:
                break



if __name__ == "__main__":
    main()