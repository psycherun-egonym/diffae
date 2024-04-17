import argparse
import json
import os

from templates import *

def parse_args():
    parser = argparse.ArgumentParser(description='Script to generate latent codes')
    parser.add_argument('--data_dir', type=str, default='imgs/')
    parser.add_argument('--output_dir', type=str, default='latent_codes/')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    conf = ffhq256_autoenc()
    print(conf.name)

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
    
    latent_codes = []
    for image in tqdm(data):
        batch = image['img'][None]
        latent_code = model.encode(batch.to(device)).cpu()

        latent_codes.append({
            "file_name": image['file_name'],
            "latent_code": latent_code.numpy().tolist()
        }) 

    with open(os.path.join(args.output_dir, 'metadata.jsonl'), 'w') as fp:
        json.dump(latent_codes, fp, indent='')   

    
if __name__ == '__main__':
    main()


