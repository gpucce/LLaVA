# pylint: disable=import-error

import json
from pathlib import Path
from argparse import ArgumentParser


import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import faiss

from tqdm.auto import tqdm

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ref_img_path', type=str, required=True)
    parser.add_argument('--img_path', type=str, required=True)
    # parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--pool_type", type=str, choices=["cls", "mean"])
    parser.add_argument("--output_path", type=str, default="output.jsonl")
    return parser.parse_args()

def get_image_paths(img_path, max_images):
    img_paths = Path(img_path).rglob('*.jpg')
    img_paths = list(img_paths)
    if max_images is not None and max_images > 0:
        img_paths = img_paths[:max_images]
    return img_paths

def create_embeddings(img_paths, processor, model, batch_size, pool_type="mean"):
    model.to('cuda')
    outputs = []
    batched_img_paths = [img_paths[i:i+batch_size] for i in range(0, len(img_paths), batch_size)]
    for image_paths in tqdm(batched_img_paths):
        images = [Image.open(image_path) for image_path in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.inference_mode():
            _outputs = model(**inputs.to('cuda'))
            _outputs = _outputs.last_hidden_state.detach().cpu()
        
        if pool_type == "cls":
            outputs.append(_outputs[:, 0])
        elif pool_type == "mean":
            outputs.append(_outputs[:, 1:].mean(dim=1).cpu())
        else:
            raise ValueError(f"Invalid pool type: {pool_type}")

    return torch.cat(outputs, dim=0).numpy().astype('float32')

def main(args):
    ref_img_paths = get_image_paths(args.ref_img_path, args.max_images)
    img_paths = get_image_paths(args.img_path, args.max_images)

    processor = AutoImageProcessor.from_pretrained(
        '/leonardo_scratch/large/userexternal/gpuccett/models/hf_vit/dinov2_giant')
    model = AutoModel.from_pretrained(
        '/leonardo_scratch/large/userexternal/gpuccett/models/hf_vit/dinov2_giant')
    model.eval()

    ref_embeddings = create_embeddings(
        ref_img_paths, processor, model, batch_size=args.batch_size, pool_type=args.pool_type)
    assert ref_embeddings.shape[0] == len(ref_img_paths)
    img_embeddings = create_embeddings(
        img_paths, processor, model, batch_size=args.batch_size, pool_type=args.pool_type)
    assert img_embeddings.shape[0] == len(img_paths)
    assert ref_embeddings.shape[1] == img_embeddings.shape[1]
    print("OUTPUT SHAPE", ref_embeddings.shape)


    faiss.normalize_L2(ref_embeddings)
    faiss.normalize_L2(img_embeddings)

    index = faiss.IndexFlatIP(ref_embeddings.shape[1]) # build the index
    print(index.is_trained)
    index.add(ref_embeddings) # add vectors to the index
    print(index.ntotal)

    k = 4 # we want to see 4 nearest neighbors
    D, I = index.search(img_embeddings, k) # actual search

    output_path = args.output_path
    output_path = (
        output_path
        .replace(".jsonl", f"pooling_{args.pool_type}.jsonl")
    )
    with open(output_path, 'w') as f:
        for idx, i in enumerate(img_paths):
            output_dict = {}
            ref_path = str(i.resolve())
            matches = np.array(ref_img_paths)[I[idx]]
            matches_paths = [str(i.resolve()) for i in matches]
            print(ref_path)
            for dist, m_p in zip(D[idx], matches_paths):
                print("\t", m_p, dist)
            output_dict[ref_path] = matches_paths
            
            f.write(json.dumps(output_dict) + "\n")

    D, I = index.search(ref_embeddings[:5], k) # sanity check
    for idx, i in enumerate(ref_img_paths[:5]):
        print(str(i.resolve()))
        matches = np.array(ref_img_paths)[I[idx]]
        for dist, m_p in zip(D[idx], matches):
            print("\t", str(m_p.resolve()), dist)


if __name__ == '__main__':
    main(parse_args())
