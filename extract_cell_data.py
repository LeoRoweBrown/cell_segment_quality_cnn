import tifffile as tiff
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import os
import re
import napari
import cv2

def search_for_runs(path):
    print("searching", f'{path}/*/*/cst*/run_*')
    run_dirs = glob(f'{path}/*/*/cst*/run_*', recursive=True)
    print(f"{len(run_dirs)} run dirs found in data:")
    [print(rdir) for rdir in run_dirs]
    return run_dirs

def search_for_mask_runs(path):
    mask_run_dirs = glob(f'{path}/run_*')
    run_nums = [int(re.findall('run_([0-9]+)', s)[0]) for s in mask_run_dirs]
    idx = np.argsort(run_nums)
    mask_run_dirs = [mask_run_dirs[i] for i in idx]
    print(f"{len(mask_run_dirs)} mask run dirs found in data:")
    [print(rdir) for rdir in mask_run_dirs]
    return mask_run_dirs

def get_padding(target_size, roi_size):
    pad = int(max((target_size - roi_size) // 2, 0))
    pad_a = [pad, (target_size - roi_size - pad)]
    return pad_a

def get_rois_fixed_roi(mask, data, roi_size=(64,64,64), show_plot=False, bg=None):
    segment_vals = np.max(mask)
    if bg is None:
        bg_vals = data[mask == 0]
        bg = np.mean(bg_vals) - np.std(bg_vals)
    data = np.maximum(data.astype(np.int32) - bg, 0).astype(np.uint16)
    # bg_std = np.std(data[mask == 0])

    data_size = data.shape
    print("bg computed to be:", bg)
    print(segment_vals, "segments")
    mask_rois = []
    data_rois = []
    data_rois_masked = []

    for i in range(1, segment_vals+1):
        # get idxs that mark current segment
        # print("segment with value", i, "size is", np.sum(mask==i))
        zi, yi, xi = np.where(mask==i)
        # min and max indices bounding segmentation region
        # print("segment", i, "of", segment_vals)
        
        if len(zi) == 0:
            print("segment with value", i, "is missing from mask!")
            continue
        min_xi, min_yi, min_zi = np.min(xi), np.min(yi), np.min(zi)
        max_xi, max_yi, max_zi = np.max(xi), np.max(yi), np.max(zi)

        # size bound by the segmentation region
        roi_min_size = np.max(zi) - np.min(zi) + 1, np.max(yi) - np.min(yi) + 1, np.max(xi) - np.min(xi) + 1
        # get padding to make roi fill the roi_size
        padx = get_padding(roi_size[2], roi_min_size[2])
        pady = get_padding(roi_size[1], roi_min_size[1])
        padz = get_padding(roi_size[0], roi_min_size[0])

        # make sure with padding min x,y,z and max x,y,z are not beyond the edge of volume
        min_zi -= min(0, min_zi-padz[0])
        min_yi -= min(0, min_yi-pady[0])
        min_xi -= min(0, min_xi-padx[0])
        max_zi -= min(0, data_size[0]-max_zi+padz[1])
        max_yi -= min(0, data_size[1]-max_yi+pady[1])
        max_xi -= min(0,data_size[2]-max_xi+padx[1])

        # finally get the rois, important to make a copy because mask_roi[mask_roi != i] = 0 would edit mask itself 
        mask_roi = mask[(min_zi-padz[0]):(max_zi+padz[1]+1), (min_yi-pady[0]):(max_yi+pady[1]+1), (min_xi-padx[0]):(max_xi+padx[1])+1].copy()
        mask_roi[mask_roi != i] = 0  # make binary, 0: background 1: current segment
        mask_roi[mask_roi == i] = 1  # reject other segments
        data_roi = data[(min_zi-padz[0]):(max_zi+padz[1]+1), (min_yi-pady[0]):(max_yi+pady[1]+1), (min_xi-padx[0]):(max_xi+padx[1])+1].copy()
        data_roi_overlay = np.array(data_roi)  # make copy
        data_roi_overlay *= mask_roi

        # final padding
        pad = np.array(roi_size) - data_roi.shape
        mask_roi = np.pad(mask_roi, ((0, pad[0]), (0, pad[1]), (0, pad[2])))
        data_roi = np.pad(data_roi, ((0, pad[0]), (0, pad[1]), (0, pad[2])))
        data_roi_overlay = np.pad(data_roi_overlay, ((0, pad[0]), (0, pad[1]), (0, pad[2])))
        mask_rois.append(mask_roi)
        data_rois.append(data_roi)
        data_rois_masked.append(data_roi_overlay)

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(data_roi[int(data_roi.shape[0]//2), :, :], vmin=0, vmax=400)
            ax2.imshow(mask_roi[int(data_roi.shape[0]//2), :, :])
            plt.show()
    return mask_rois, data_rois, data_rois_masked


def get_rois(mask, data, show_plot=False):
    segment_vals = np.max(mask)
    bg_std = np.std(data[mask == 0])
    bg = np.mean(data[mask == 0]) - bg_std
    data -= bg.astype(np.uint16)
    print("bg", bg)
    mask_rois = []
    data_rois = []
    for i in range(1, segment_vals):
        segment_mask = (mask == i)
        xi, yi, zi = np.where(mask==i)

        roi_size = np.max(xi) - np.min(xi) + 1, np.max(yi) - np.min(yi) + 1, np.max(zi) - np.min(zi) + 1
        roi_x_idx = range(np.min(xi), np.max(xi))
        roi_y_idx = range(np.min(yi), np.max(yi))
        roi_z_idx = range(np.min(zi), np.max(zi))

        min_xi, min_yi, min_zi = np.min(xi), np.min(yi), np.min(zi)
        max_xi, max_yi, max_zi = np.max(xi), np.max(yi), np.max(zi)

        mask_roi = np.zeros(roi_size)
        data_roi = np.zeros(roi_size)
        mask_roi[xi - min_xi, yi - min_yi, zi - min_zi] = mask[xi, yi, zi]
        data_roi[xi - min_xi, yi - min_yi, zi - min_zi] = data[xi, yi, zi]

        mask_rois.append(mask_roi)
        data_rois.append(data_roi)

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(data_roi[:, :, int(data_roi.shape[0]//2)], vmin=0, vmax=600)
            ax2.imshow(mask_roi[:, :, int(data_roi.shape[0]//2)])
            plt.show()
    return mask_rois, data_rois


def save_rois(save_path, mask_rois, data_rois, data_rois_masked=None):
    print(f"saving roi tiff stacks in {save_path}")
    os.makedirs(save_path, exist_ok=True)
    for n in range(len(mask_rois)):
        save_path_mask = os.path.join(save_path, f'mask_roi_{n:03d}.tif')
        save_path_data = os.path.join(save_path, f'data_roi_{n:03d}.tif')
        save_path_data_masked = os.path.join(save_path, f'data_roi_masked_{n:03d}.tif')
        tiff.imwrite(save_path_mask, mask_rois[n], metadata={'axes': 'ZYX'})
        tiff.imwrite(save_path_data, data_rois[n], metadata={'axes': 'ZYX'})
        if data_rois_masked is not None:
           tiff.imwrite(save_path_data_masked, data_rois_masked[n], metadata={'axes': 'ZYX'}) 

def get_projections(img, project_mode='max', slice_idx=[None,None,None]):
    for n in range(len(slice_idx)):
        slice_idx[n] = int(img.shape[n]//2) if slice_idx[n] is None else slice_idx[n]
    if project_mode=='max':
        xy = np.max(img, 0)
        xz = np.max(img, 1)
        yz = np.max(img, 2)
    elif project_mode=='com_slice':
        xy = img[slice_idx[0], :, :]
        xz = img[:, slice_idx[1], :]
        yz = img[:, :,  slice_idx[2]]
    else:
        xy = img[int(img.shape[0]//2), :, :]
        xz = img[:, int(img.shape[1]//2), :]
        yz = img[:, :,  int(img.shape[2]//2)]
    return [xy, xz, yz]

def save_plots(save_path, mask_rois, data_rois, data_rois_masked=None, project_mode='com_slice', draw_outlines=True, vmax=350):
    print(f"saving plots in {save_path}")
    os.makedirs(save_path, exist_ok=True)
    vmax_ = vmax
    for n in range(len(mask_rois)):
        save_path_r = os.path.join(save_path, f'roi_{n:03d}.png')

        if data_rois_masked is not None:
            fig, ((ax1, ax2, ax1b, ax2b, ax1c, ax2c), (ax3, ax4, ax3b, ax4b, ax3c, ax4c)) = plt.subplots(2,6, layout='compressed')
            ax4c.set_visible(False)
        else:
            fig, ((ax1, ax2, ax1b, ax2b), (ax3, ax4, ax3b, ax4b)) = plt.subplots(2,4, layout='compressed')
        fig.set_tight_layout(True)
        ax4.set_visible(False)
        ax4b.set_visible(False)

        slice_idx = [None, None, None]
        mask_roi = mask_rois[n]
        if project_mode == 'com_slice':
            z_indices, y_indices, x_indices = np.indices(mask_roi.shape)  # 3d matrices of indices a bit like meshgrid but 1d
            total_mass = np.sum(mask_roi)
            x_cm = int(np.round(np.sum(x_indices * mask_roi) / total_mass))
            y_cm = int(np.round(np.sum(y_indices * mask_roi) / total_mass))
            z_cm = int(np.round(np.sum(z_indices * mask_roi) / total_mass))
            slice_idx = [z_cm, y_cm, x_cm]
        mask_xy, mask_xz, mask_yz = get_projections(mask_roi, project_mode=project_mode, slice_idx=slice_idx)

        ax1.imshow(mask_xy, vmin=0, vmax=1)
        ax2.imshow(mask_xz, vmin=0, vmax=1)
        ax3.imshow(mask_yz, vmin=0, vmax=1)
        
        # get outlines from mask
        contours_xy, _ = cv2.findContours(mask_xy.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_xz, _ = cv2.findContours(mask_xz.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yz, _ = cv2.findContours(mask_yz.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        data_roi = data_rois[n]
        data_xy, data_xz, data_yz = get_projections(data_roi, project_mode=project_mode, slice_idx=slice_idx)

        if vmax == 0 or vmax is None:
            vmax_ = (np.max(data_xy) + np.max(data_xz) + np.max(data_yz)) / 3
            # print("auto vmax =", vmax_)
        if draw_outlines:
            # values that equal vmax -> 255, so vals/vmax then * 255
            data_xy = np.clip(np.asarray(data_xy, dtype=np.float64)*(255/vmax_), 0, 255)  # convert to 8bit
            data_xz = np.clip(np.asarray(data_xz, dtype=np.float64)*(255/vmax_), 0, 255)
            data_yz = np.clip(np.asarray(data_yz, dtype=np.float64)*(255/vmax_), 0, 255)
            # print(data_xy)
            data_xy = cv2.cvtColor(np.asarray(data_xy, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            data_xz = cv2.cvtColor(np.asarray(data_xz, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            data_yz = cv2.cvtColor(np.asarray(data_yz, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(data_xy, contours_xy, -1, [255,0,0], 1)
            cv2.drawContours(data_xz, contours_xz, -1, [255,0,0], 1)
            cv2.drawContours(data_yz, contours_yz, -1, [255,0,0], 1)

            ax1b.imshow(data_xy, vmax=255)
            ax2b.imshow(data_xz, vmax=255)
            ax3b.imshow(data_yz, vmax=255)

        ax1b.imshow(data_xy, vmax=vmax_)
        ax2b.imshow(data_xz, vmax=vmax_)
        ax3b.imshow(data_yz, vmax=vmax_)

        # Plot the masked cells
        if data_rois_masked is not None:
            data_roi_m = data_rois_masked[n]
            data_m_xy, data_m_xz, data_m_yz = get_projections(data_roi_m, project_mode=project_mode, slice_idx=slice_idx)

            ax1c.imshow(data_m_xy, vmax=vmax)
            ax2c.imshow(data_m_xz, vmax=vmax)
            ax3c.imshow(data_m_yz, vmax=vmax)


        fig.savefig(save_path_r)
        plt.close()
            

#----------------------------Script start-----------------------------------------

if __name__ == '__main__':
    save_dir = "G:/Data/IBIN_Nina/workspace/cnn_model/data/"
    save_plot_dir = "G:/Data/IBIN_Nina/workspace/cnn_model/data_plots/"

    channel = 'exc561_filter605'
    data_dir = "G:/Data/IBIN_Nina/temp/20230331_drug_plate3/main/"
    run_dirs = search_for_runs(data_dir)
    mask_dir = "G:/Data/IBIN_Nina/workspace/trained_masks_drugplate3/all_runs_vols_cellpose_reslice_150epoch"
    mask_run_dirs = glob(f'{mask_dir}/run_*')
    run_sub_dirs = [os.path.basename(run_dir) for run_dir in mask_run_dirs]
    mask_field_files = glob(f'{mask_run_dirs[0]}/field*')  # assumes same fields in every run folder!
    run_nums = [int(re.findall('run_([0-9]+)', s)[0]) for s in run_sub_dirs]
    field_nums = [int(re.findall('field([0-9]+)*', s)[0]) for s in mask_field_files]
    mask_field_files

    all_fields = False
    all_runs = False

    runs_to_include = [1, 5, 11]
    fields_to_include = [1]#, 150]

    if all_fields:
        fields_to_include = field_nums

    if all_runs:
        runs_to_include = run_nums

    print(f"Extracting cells for runs {runs_to_include} and fields {fields_to_include}")

    for n in runs_to_include:
        data_run_dir = run_dirs[n-1]
        mask_run_dir = mask_run_dirs[n-1]
        for f in fields_to_include:
            mask_file = os.path.join(mask_run_dir, os.path.basename(mask_field_files[f-1]))
            data_file = os.path.join(data_run_dir, f'field_{f:04d}', channel, '*.tif')
            print("loading", mask_file, "and", data_file)
            mask = tiff.imread(mask_file)
            data = tiff.imread(data_file)
            # viewer=napari.Viewer(show=True)
            viewer, image_layer = napari.imshow(data)
            napari.run()  # Starts the Qt event loop (blocks here)
            # viewer.add_image(data)
            # input("Press Enter to continue...")
            mask_rois, data_rois, data_rois_masked = get_rois_fixed_roi(mask, data, (32, 32, 32), bg=98)
            save_path = os.path.join(save_dir, f'run_{n:04d}_field_{f:04d}')
            save_path_dir = os.path.join(save_plot_dir, f'run_{n:04d}_field_{f:04d}')
            save_rois(save_path, mask_rois, data_rois, data_rois_masked)
            save_plots(save_path_dir, mask_rois, data_rois, data_rois_masked)

