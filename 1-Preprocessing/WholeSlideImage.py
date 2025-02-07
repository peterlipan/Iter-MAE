import os
import cv2
import h5py
import math
import time
import pathlib
import logging
import openslide
import numpy as np
from PIL import Image
from typing import Union
import multiprocessing as mp
from skimage.filters import threshold_otsu
from dataset_helpers import Pool


class isInContourV3_Easy:
	def __init__(self, contour, patch_size, center_shift=0.5):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0


class WholeSlideImage(object):
    def __init__(self, src: str, dst: str, patch_size: str=512, base_downsample: int=1,
                 use_otsu: bool=True, sthresh: int=8, sthresh_up: int=255, mthresh: int=7, padding: bool=True, visualize: bool=True,
                 visualize_width: int=1024, skip: bool=True, save_patch: bool=False, style: str='DTFD'):
        self.src = src
        self.dst = dst
        self.patch_size = patch_size

        self.use_otsu = use_otsu
        self.sthresh = sthresh
        self.sthresh_up = sthresh_up
        self.mthresh = mthresh

        self.wsi_name = pathlib.Path(src).stem
        self.wsi = openslide.OpenSlide(src)
        self.level_count = self.wsi.level_count
        self.level_dimensions = self.wsi.level_dimensions
        self.level_downsamples = self._assertLevelDownsamples()
        self.base_downsample = base_downsample
        if base_downsample not in self.wsi.level_downsamples:
            self.base_level = self.wsi.get_best_level_for_downsample(base_downsample)
            self.vis_level = self.wsi.get_best_level_for_downsample(64)
            cur_downsample = self.wsi.level_downsamples[self.base_level]
            logging.warning(f'Base downsample {base_downsample} not available for {self.wsi_name}. Using downsample {cur_downsample} instead.')
        else:
            self.base_level = self.wsi.get_best_level_for_downsample(base_downsample)
            self.vis_level = self.wsi.get_best_level_for_downsample(64)
        self.base_dimensions = self.level_dimensions[self.base_level]
        self.padding = padding
        self.visualize = visualize
        self.visualize_width = visualize_width
        self.skip = skip
        self.save_patch = save_patch
        self.style = style
        self.palette = [(173, 216, 230, 255), (255, 182, 193, 255), (152, 251, 152, 255), (230, 230, 250, 255),
                        (255, 255, 0, 255), (255, 165, 0, 255), (255, 0, 255, 255), (64, 224, 208, 255),
                        (168, 168, 120, 255), (210, 105, 30, 255), (255, 199, 0, 255), (138, 54, 15, 255)]

    def _assertLevelDownsamples(self):
        # estimate the downsample factor for each level, following CLAM
        level_downsamples = []
        dim_0 = self.level_dimensions[0]

        for downsample, dim in zip(self.wsi.level_downsamples, self.wsi.level_dimensions):
            estimated_downsample = (dim_0[0] / float(dim[0]), dim_0[1] / float(dim[1]))

            level_downsamples.append(estimated_downsample) if estimated_downsample != (
                downsample, downsample) else level_downsamples.append((downsample, downsample))

        return level_downsamples


    def visWSI(self, color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
                    line_thickness=500, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                    number_contours=False, seg_display=True, annot_display=True):
        
        downsample = self.level_downsamples[self.vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dimensions[self.vis_level]

        img = np.array(self.wsi.read_region(top_left, self.vis_level, region_size).convert("RGB"))
        
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_tissue is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_tissue):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center
                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                for holes in self.holes_tissue:
                    cv2.drawContours(img, self.scaleContourDim(holes, scale), 
                                     -1, hole_color, line_thickness, lineType=cv2.LINE_8)
            
            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), 
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)
        
        img = Image.fromarray(img)
    
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

    def _visualize_segmentation(self, img, asset_dict, stop_x, stop_y):
        scale = self.level_downsamples[self.base_level][0]
        save_path = os.path.join(self.dst, 'visualization', f'{self.wsi_name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        height, width, _ = img.shape
        new_height = int(self.visualize_width * height / width)
        resized_img = cv2.resize(img, (self.visualize_width, new_height), interpolation=cv2.INTER_CUBIC)
        resized_height, resized_width, _ = resized_img.shape
        scaled_stop_x = int(stop_x / scale / width * resized_width)
        scaled_stop_y = int(stop_y / scale / height * resized_height)


        grid_x, grid_y = asset_dict['coord'][:, :, 0], asset_dict['coord'][:, :, 1]
        scaled_grid_x = grid_x[:, 0] / scale / width * resized_width
        scaled_grid_y = grid_y[0] / scale / height * resized_height

        scaled_start_x = int(min(scaled_grid_x))
        scaled_start_y = int(min(scaled_grid_y))

        for x in set(scaled_grid_x):
            cv2.line(resized_img, (int(x), scaled_start_y), (int(x), scaled_stop_y-1), self.palette[0], 2)

        for y in set(scaled_grid_y):
            cv2.line(resized_img, (scaled_start_x, int(y)), (scaled_stop_x-1, int(y)), self.palette[0], 2)

        # draw the end line
        cv2.line(resized_img, (scaled_stop_x-1, scaled_start_y), (scaled_stop_x-1, scaled_stop_y-1), self.palette[0], 2)
        cv2.line(resized_img, (scaled_start_x, scaled_stop_y-1), (scaled_stop_x-1, scaled_stop_y-1), self.palette[0], 2)

        cv2.imwrite(save_path, resized_img)
    
    @staticmethod
    def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):
        if WholeSlideImage.isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
            return coord
        else:
            return None

    @staticmethod
    def isInHoles(holes, pt, patch_size):
        for hole in holes:
            if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
                return 1
        
        return 0

    @staticmethod
    def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
        if cont_check_fn(pt):
            if holes is not None:
                return not WholeSlideImage.isInHoles(holes, pt, patch_size)
            else:
                return 1
        return 0

    @staticmethod
    def save_hdf5(output_path, asset_dict, attr_dict, mode='a'):
        file = h5py.File(output_path, mode)
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape,
                                           dtype=data_type)
                dset[:] = val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
        for key, val in attr_dict.items():
            file.attrs[key] = val
        file.close()
        return output_path

    def segment(self):
        h5_path = os.path.join(self.dst, 'coordinates', f'{self.wsi_name}.h5')
        if os.path.exists(h5_path) and self.skip:
            print(f'\n{self.wsi_name} already processed. Skipping...')
            logging.info(f'{self.wsi_name} already processed. Skipping...')
            return
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)

        # load the WSI
        print(f'loading {self.wsi_name}...')
        start = time.time()
        img = np.array(self.wsi.read_region((0, 0), self.base_level, self.base_dimensions))
        print(f'WSI loaded in {time.time() - start:.2f}s')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # following CLAM
        if self.style == 'CLAM':
            img_med = cv2.medianBlur(img_hsv[:, :, 1], self.mthresh)

            # thresholding
            if self.use_otsu:
                print('Using Otsu thresholding')
                _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY)

            # the minimum bounding box of the whole tissue
            contours, _ = cv2.findContours(img_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.concatenate(contours)
            # scale the coord to level 0
            scale = self.level_downsamples[self.base_level]
            contours = (contours * scale).astype(np.int32)

        elif self.style == 'DTFD':
            # DTFD's way of preprocessing
            h, s, v = cv2.split(img_hsv)

            hthresh = threshold_otsu(h)
            sthresh = threshold_otsu(s)
            vthresh = threshold_otsu(v)

            minhsv = np.array([hthresh, sthresh, 70], np.uint8)
            maxhsv = np.array([180, 255, vthresh], np.uint8)
            thresh = [minhsv, maxhsv]
            mask = cv2.inRange(img_hsv, thresh[0], thresh[1])

            close_kernel = np.ones((100, 100), dtype=np.uint8)
            image_close_img = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
            open_kernel = np.ones((60, 60), dtype=np.uint8)
            image_open_np = cv2.morphologyEx(np.array(image_close_img), cv2.MORPH_OPEN, open_kernel)

            contours, _ = cv2.findContours(image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   #_, contours, _ = cv2.findContours(image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.concatenate(contours)
            scale = self.level_downsamples[self.base_level]
            contours = (contours * scale).astype(np.int32)

        else:
            raise ValueError(f'Unknown style: {self.style}')

        x, y, w, h = cv2.boundingRect(contours)

        img_w, img_h = self.level_dimensions[0]
        base_patch_size = self.patch_size * scale[0]

        if self.padding:
            stop_y = y + h
            stop_x = x + w
        else:
            # drop the last patch if it is smaller than the patch size
            stop_y = min(y + h, img_h - base_patch_size + 1)
            stop_x = min(x + w, img_w - base_patch_size + 1)

        print("Bounding box: ", x, y, w, h)
        print("Contour area: ", cv2.contourArea(contours))

        # No need to check the holes. Directly generate the mesh
        asset_dict = {}

        step_size = int(base_patch_size )
        x_range = np.arange(x, stop_x, step_size)
        y_range = np.arange(y, stop_y, step_size)
        x_coord, y_coord = np.meshgrid(x_range, y_range, indexing='ij')
        asset_dict['coord'] = np.stack([x_coord, y_coord], axis=-1)
    
        # For faster downstream feature extraction, directly resize and save the patches
        # use the same reading method as in the feature extraction to ensure it is correct
        if self.save_patch:
            patch_path = os.path.join(self.dst, 'patches', f'{self.wsi_name}')
            os.makedirs(patch_path, exist_ok=True)
        
            coords = asset_dict['coord']
            
            for m in range(coords.shape[0]):
                for n in range(coords.shape[1]):
                    x, y = coords[m, n]
                    patch = np.array(self.wsi.read_region((int(x), int(y)), self.base_level, (self.patch_size, self.patch_size)))
                    cv2.imwrite(os.path.join(patch, f'{m}_{n}_.png'), patch)
        

        if self.visualize:
            self._visualize_segmentation(img, asset_dict, stop_x, stop_y)

        attr_dict = {'base_level': self.base_level, 'base_dimensions': self.base_dimensions,
                     'base_downsample': self.base_downsample, 'padding': self.padding,
                     'patch_size': self.patch_size,}

        assert asset_dict, "Asset dictionary is empty"

        self.save_hdf5(h5_path, asset_dict, attr_dict, mode='w')

    def segment_tissue(self):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params):
            """
                Filter contours by: area.
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []
            
            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)


            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours
        
        start = time.time()
        img = np.array(self.wsi.read_region((0, 0), self.base_level, self.base_dimensions))
        print(f'WSI loaded in {time.time() - start:.2f}s')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8} # default values from CLAM
        
        if self.style == 'CLAM':
            print('Using CLAM preprocessing')
            
            img_med = cv2.medianBlur(img_hsv[:,:,1], self.mthresh)  # Apply median blurring
            _, img_otsu = cv2.threshold(img_med, self.sthresh, self.sthresh_up, cv2.THRESH_BINARY)
            kernel = np.ones((4, 4), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)     

        elif self.style == 'DTFD':
            print('Using DTFD preprocessing')
            h, s, v = cv2.split(img_hsv)

            hthresh = threshold_otsu(h)
            sthresh = threshold_otsu(s)
            vthresh = threshold_otsu(v)

            minhsv = np.array([hthresh, sthresh, 70], np.uint8)
            maxhsv = np.array([180, 255, vthresh], np.uint8)
            thresh = [minhsv, maxhsv]
            mask = cv2.inRange(img_hsv, thresh[0], thresh[1])

            close_kernel = np.ones((100, 100), dtype=np.uint8)
            image_close_img = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
            open_kernel = np.ones((60, 60), dtype=np.uint8)
            img_otsu = cv2.morphologyEx(np.array(image_close_img), cv2.MORPH_OPEN, open_kernel)


        scale = self.level_downsamples[self.base_level]
        ref_patch_size = self.patch_size * scale[0]
        scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))
        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
        
        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        if filter_params: 
            foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

        self.contours_tissue = self.scaleContourDim(foreground_contours, scale)
        self.holes_tissue = self.scaleHolesDim(hole_contours, scale)

    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 256, step_size = 256,
        top_left=None, bot_right=None):
        if cont is not None:
             start_x, start_y, w, h = cv2.boundingRect(cont)
        else:
            start_x, start_y, w, h = 0, 0, self.level_dimensions[patch_level][0], self.level_dimensions[patch_level][1]

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        
        img_w, img_h = self.level_dimensions[self.base_level]
        if self.padding:
            stop_y = start_y+h
            stop_x = start_x+w
        else:
            stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
            stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)
        
        print("Bounding Box:", start_x, start_y, w, h)
        print("Contour Area:", cv2.contourArea(cont))

        if bot_right is not None:
            stop_y = min(bot_right[1], stop_y)
            stop_x = min(bot_right[0], stop_x)
        if top_left is not None:
            start_y = max(top_left[1], start_y)
            start_x = max(top_left[0], start_x)

        if bot_right is not None or top_left is not None:
            w, h = stop_x - start_x, stop_y - start_y
            if w <= 0 or h <= 0:
                print("Contour is not in specified ROI, skip")
                return {}, {}
            else:
                print("Adjusted Bounding Box:", start_x, start_y, w, h)
    

        cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size[0], center_shift=0.5)

        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]

        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        num_workers = mp.cpu_count()
        if num_workers > 4:
            num_workers = 4
        pool = Pool(num_workers)

        iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
        results = pool.starmap(WholeSlideImage.process_coord_candidate, iterable)
        pool.close()
        pool.join()
        results = np.array([result for result in results if result is not None])
        
        print('Extracted {} coordinates'.format(len(results)))
        print("patch_size: ", patch_size)
        print("patch_level: ", patch_level)
        print("downsample: ", self.level_downsamples[patch_level])
        print("level_dimensions: ", self.level_dimensions[patch_level])
        print("wsi_name: ", self.wsi_name)
        print("save_path: ", save_path)

        if len(results)>0:
            asset_dict = {'coords' :          results}
            
            attr = {'patch_size' :            patch_size, # To be considered...
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'level_dimensions':       self.level_dimensions[patch_level],
                    'wsi_name':               self.wsi_name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
            return asset_dict, attr_dict

        else:
            return {}, {}

    def patchify(self):
        save_path_hdf5 = os.path.join(self.dst, 'coordinates', f'{self.wsi_name}.h5')
        if os.path.exists(save_path_hdf5) and self.skip:
            print(f'\n{self.wsi_name} already processed. Skipping...')
            logging.info(f'{self.wsi_name} already processed. Skipping...')
            return
        os.makedirs(os.path.dirname(save_path_hdf5), exist_ok=True)
        print("Creating patches for: ", self.wsi_name, "...",)
        self.segment_tissue()
        n_contours = len(self.contours_tissue)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_tissue):
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            
            asset_dict, attr_dict = self.process_contour(cont, self.holes_tissue[idx], self.base_level, save_path_hdf5, self.patch_size, self.patch_size)
            if len(asset_dict) > 0:
                if init:
                    self.save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    self.save_hdf5(save_path_hdf5, asset_dict, mode='a')
        
        if self.visualize:
            vis_save_path = os.path.join(self.dst, 'visualization', f'{self.wsi_name}.png')
            os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
            mask = self.visWSI()
            mask.save(vis_save_path)
