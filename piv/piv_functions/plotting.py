"""
Visualization functions for PIV analysis.

This module contains functions for creating plots and visualizations
of PIV displacement and velocity data.
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from tqdm import trange
import sys

# Add the functions directory to the path and import CVD check
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'functions'))
import cvd_check as cvd

from .utils import get_time
from .io import save_cfig


def plot_vel_comp(disp_glo, disp_nbs, disp_spl, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array using helper function
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp_glo.shape[0] != time.shape[0]:
        disp_glo = disp_glo[frs[0]:frs[-1], :, :, :]
        disp_nbs = disp_nbs[frs[0]:frs[-1], :, :, :]
        disp_spl = disp_spl[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel_glo = disp_glo * res / dt
    vel_nbs = disp_nbs * res / dt
    vel_spl = disp_spl * res / dt

    # Scatter plot vx(t)
    fig, ax = plt.subplots(figsize=(10, 6))

    # ax.plot(np.tile(time[:, None] * 1000, (1, n_peaks)).flatten(),
    #         vel_unf[:, 0, 0, :, 1].flatten(), 'x', c='gray', alpha=0.5, ms=4, label='vx (all candidate peaks)')
    # ax.plot(1000 * time, vel_unf[:, 0, 0, 0, 1].flatten(), 'x', c='gray', alpha=0.5, ms=4, label='vx (brightest peak)')

    ax.plot(1000 * time, vel_glo[:, 0, 0, 1], 'o', ms=4, c='gray',
            label='vx (filtered globally)')
    ax.plot(1000 * time, vel_nbs[:, 0, 0, 1], '.', ms=2, c='black',
            label='vx (filtered neighbours)')
    ax.plot(1000 * time, vel_nbs[:, 0, 0, 0], c=cvd.get_color(0), 
            label='vy (filtered neighbours)')
    ax.plot(1000 * time, vel_spl[:, 0, 0, 1], c=cvd.get_color(1),
        label='vx (smoothed for 2nd pass)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('First pass')
    ax.set(**kwargs)

    ax.legend()
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_med(disp, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp.shape[0] != time.shape[0]:
        disp = disp[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel = disp * res / dt

    # Plot the median velocity in time, show the min and max as a shaded area
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot vy (vertical velocity)
    ax.plot(time * 1000,
            np.nanmedian(vel[:, :, :, 0], axis=(1, 2)), label='Median vy')
    ax.fill_between(time * 1000,
                    np.nanmin(vel[:, :, :, 0], axis=(1, 2)),
                    np.nanmax(vel[:, :, :, 0], axis=(1, 2)),
                    alpha=0.3, label='Min/max vy')

    # Plot vx (horizontal velocity)
    ax.plot(time * 1000,
            np.nanmedian(vel[:, :, :, 1], axis=(1, 2)), label='Median vx')
    ax.fill_between(time * 1000,
                    np.nanmin(vel[:, :, :, 1], axis=(1, 2)),
                    np.nanmax(vel[:, :, :, 1], axis=(1, 2)),
                    alpha=0.3, label='Min/max vx')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Median velocity in time')
    ax.set(**kwargs)

    ax.legend()
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_prof(disp, res, frs, dt, win_pos, 
                  mode="random", proc_path=None, file_name=None, subfolder=None, test_mode=False, **kwargs):
    # TODO: Write docstring
    
    # Define a time array
    n_corrs = disp.shape[0]
    time = get_time(frs, dt)
    
    # Convert displacement to velocity
    vel = disp * res / dt
  
    # Raise error if one tries to make a video, but proc_path is not specified
    if mode == "video" and (proc_path is None or file_name is None
                             or test_mode):
        raise ValueError("proc_path and file_name must be specified, and test_mode must be False to create a video.")

    # Set up save path if subfolder is specified
    if proc_path is not None and subfolder is not None and not test_mode:
        save_path = os.path.join(proc_path, subfolder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        if mode == "all":
            # Error: we don't want to save all images to the root folder
            raise RuntimeWarning(f"Are you sure you want to save {n_corrs} files directly to {proc_path}?")
        save_path = proc_path
    
    # Determine which frames to process
    if mode == "random":
        np.random.seed(42)  # For reproducible results
        frames_to_plot = np.sort(np.random.choice(n_corrs, size=min(10, n_corrs), replace=False))
    elif mode == "all" or mode == "video":
        frames_to_plot = range(n_corrs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'video', 'all', or 'random'.")
    
    # Set up video writer if needed
    if mode == "video":
        fig_video, ax_video = plt.subplots(figsize=(10, 6))
        writer = ani.FFMpegWriter(fps=10)
        video_path = os.path.join(proc_path, file_name+'.mp4')
        video_context = writer.saving(fig_video, video_path, dpi=150)
        frames_iter = trange(n_corrs, desc='Creating velocity profile video')
    else:
        video_context = None
        frames_iter = frames_to_plot
    
    # Common plotting function
    def plot_frame(frame_idx, ax):
        y_pos = win_pos[:, 0, 0] * res * 1000
        vx = vel[frame_idx, :, 0, 1]
        vy = vel[frame_idx, :, 0, 0]
        
        ax.plot(vx, y_pos, '-o', c=cvd.get_color(1), label='vx')
        ax.plot(vy, y_pos, '-o', c=cvd.get_color(0), label='vy')
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('y position (mm)')
        ax.set_title(f'Velocity profiles at frame {frame_idx + 1} ({time[frame_idx] * 1000:.2f} ms)')
        ax.legend()
        ax.grid()
        ax.set(**kwargs)
    
    # Process frames
    if video_context is not None:
        # Video mode
        with video_context:
            for i in frames_iter:
                ax_video.clear()
                plot_frame(i, ax_video)
                writer.grab_frame()
        plt.close(fig_video)
        print(f"Video saved to {video_path}")
    else:
        # Plot mode (random or all)
        for frame_idx in frames_iter:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_frame(frame_idx, ax)
            
            # Save if path is specified
            if save_path is not None:
                save_cfig(save_path, file_name + f"_{frame_idx:04d}", test_mode=test_mode)
                
                # Close figure
                plt.close(fig)
