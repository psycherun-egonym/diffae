import argparse
import json
import os

from templates import *

def parse_args():
    parser = argparse.ArgumentParser(description='Script to generate latent codes')
    parser.add_argument('--data_dir', type=str, default='../datasets/ffhq+fairface256_100k/00000/')
    parser.add_argument('--output_dir', type=str, default='../datasets/ffhq+fairface256_100k/00000/')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    conf = ffhq256_autoenc()

    # TODO: load just the semantic encoder instead of the whole model

    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    data = ImageDataset(args.data_dir, 
                        image_size=conf.img_size, 
                        exts=['jpg', 'JPG', 'png'], 
                        do_augment=False)
    print('Dataset size: ', len(data))


    latents = []
    for image in tqdm(data):
        batch = image['img'][None]
        latent_code = model.encode(batch.to(device))
        latent_code = latent_code.cpu().squeeze()
        latent_code = latent_code.numpy().tolist()

        row = {
            "file_name": image['file_name'],
            "latent_code": latent_code,
            "text": "a face of a person"
        } 
        latents.append(row)

    out_file_path = os.path.join(args.output_dir, 'metadata.jsonl')

    with open(out_file_path, 'w') as fp:
        for row in latents:
            print(json.dumps(row), file=fp)   

    # test if the written jsonl file is valid
    # with open(out_file_path, 'r') as fp:
    #     for row in fp:
    #         obj = json.loads(row)  

if __name__ == '__main__':
    main()


