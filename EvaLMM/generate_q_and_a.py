import argparse
from copy import deepcopy
from random import shuffle, seed
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images(image_files):
    images = []
    for image_file in image_files:
        images.append(load_image(image_file))
    return images


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device
    )

    seed(args.seed)
    image_paths = [str(i) for i in Path(args.image_file_path).iterdir()
                   if str(i).endswith(".jpg")]
    shuffle(image_paths)
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    batch_size = args.batch_size
    image_paths_batches = [
        image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]

    for image_paths in image_paths_batches:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(f'[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}') # pylint: disable=line-too-long
        else:
            args.conv_mode = conv_mode

        convs = [deepcopy(conv_templates[args.conv_mode].copy()) for _ in image_paths]
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = convs[0].roles


        # image = load_image(image_path)
        images = load_images(image_paths)
        image_sizes = [image.size for image in images]
        # Similar operation in model_worker.py
        image_tensor = process_images(images, image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        city = args.city.replace("_", " ")
        conv_steps = [
            "Ask me 5 question about this photo",
            "Please answer the 5 questions you asked",
            f"Please answer the 5 questions you asked knowing that the image is taken in {city}"
        ]
        image_file_names = [Path(image_path).name for image_path in image_paths]
        output_path = Path(args.output_path) / city
        output_path.mkdir(parents=True, exist_ok=True)
        for image, image_file_name in zip(images, image_file_names):
            image.save(output_path / image_file_name)

        paths = {image_file_name.replace(".jpg", ".txt") : ""
                 for image_file_name in image_file_names}

        for _inp in conv_steps:
            for image_file_name in image_file_names:
                paths[image_file_name.replace(".jpg", ".txt")] += (
                    f"{roles[0]}: \n{_inp} \n{roles[1]}: \n")

            for image, conv in zip(images, convs):
                inp = deepcopy(_inp)
                if image is not None:
                    # first message
                    if model.config.mm_use_im_start_end:
                        inp = (DEFAULT_IM_START_TOKEN +
                               DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp)
                    else:
                        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

                    conv.append_message(conv.roles[0], inp)
                else:
                    # later messages
                    conv.append_message(conv.roles[0], inp)

                conv.append_message(conv.roles[1], None)

            images = [None for _ in images]
            prompts = [conv.get_prompt() for conv in convs]

            input_ids = tokenizer_image_token(
                prompts, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).to(model.device)

            # stop_str = convs[0].sep if convs[0].sep_style != SeparatorStyle.TWO else convs[0].sep2
            # keywords = [stop_str]

            with torch.inference_mode():
                print(input_ids.shape, image_tensor.shape, image_sizes)
                output_ids = model.generate(
                    input_ids,
                    images=[image for image in image_tensor],
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True)

            outputs = [i.strip() for i in tokenizer.batch_decode(output_ids)]
            for conv, output, image_file_name in zip(convs, outputs, image_file_names):
                paths[image_file_name.replace(".jpg", ".txt")] += output + "\n"
                conv.messages[-1][-1] = output

            if args.debug:
                for prompt, output in zip(prompts, outputs):
                    print("\n", {"prompt": prompt, "outputs": output}, "\n")

        for output_name, text in paths.items():
            with open(output_path / output_name, "w", encoding="utf-8") as f:
                f.write(text)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-path", type=str, default="captioned")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
