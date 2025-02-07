import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from WholeSlideImage import WholeSlideImage
import multiprocessing as mp


def process_slide(args, slide_path, dst_dir):
    try:
        wsi = WholeSlideImage(slide_path, dst_dir, patch_size=args.patch_size, base_downsample=args.base_downsample,
                              use_otsu=not args.no_use_otsu,
                              sthresh=args.sthresh, sthresh_up=args.sthresh_up, mthresh=args.mthresh, padding=True,
                              visualize=not args.no_visualize, visualize_width=args.visualize_width, skip=not args.no_skip, save_patch=args.save_patch)
        wsi.segment()
        return slide_path, 'done'
    except Exception as e:
        print(f'Error processing {slide_path}:')
        print(e)
        return slide_path, 'error'


def init_df(args):
    image_extensions = ['.tif', '.tiff', '.svs', '.mrxs', '.ndpi']
    
    slide_paths = [f for f in Path(args.src).rglob('*') if f.suffix in image_extensions]
    
    slide_ids = [f.name for f in slide_paths]
    slide_paths_str = [str(f) for f in slide_paths]
    
    # Create status and process lists
    status = ['tbp'] * len(slide_paths_str)
    process = [1] * len(slide_paths_str)
    
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'slide_path': slide_paths_str,
        'status': status,
        'process': process
    })
    
    return df



def main(args):
    os.makedirs(args.dst, exist_ok=True)
    df_path = os.path.join(args.dst, 'status.csv')
    df = init_df(args)
    df.to_csv(df_path, index=False)

    with mp.Pool(processes=args.workers) as pool:
        results = [pool.apply_async(process_slide, args=(args, slide_path, args.dst)) for slide_path in df['slide_path']]
        pool.close()
        pool.join()
        for i, res in tqdm(enumerate(results), total=len(results)):
            slide_path, status = res.get()
            df.loc[df['slide_path'] == slide_path, 'status'] = status
        df.to_csv(df_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whole Slide Image Processing')
    parser.add_argument('--src', type=str, default='/data1/public/TCGA-GL')
    parser.add_argument('--dst', type=str, default='/data1/public/TCGA-GL_patches')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--base_downsample', type=int, default=2) # at x20
    parser.add_argument('--no_use_otsu', action='store_true')
    parser.add_argument('--sthresh', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--sthresh_up', type=int, default=255)
    parser.add_argument('--mthresh', type=int, default=7)
    parser.add_argument('--no_visualize', action='store_true')
    parser.add_argument('--visualize_width', type=int, default=1024)
    parser.add_argument('--no_skip', action='store_true')
    parser.add_argument('--save_patch', action='store_true')
    args = parser.parse_args()
    main(args)
