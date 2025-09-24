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


def plot_vel_comp(disp_glo, disp_nbs, disp_spl, res, frs, dt, proc_path=None, file_name=None, test_mode=False, 
                  disp_rejected=None, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array using helper function
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp_glo.shape[0] != time.shape[0]:
        disp_glo = disp_glo[frs[0]:frs[-1], :, :, :]
        disp_nbs = disp_nbs[frs[0]:frs[-1], :, :, :]
        disp_spl = disp_spl[frs[0]:frs[-1], :, :, :]
        if disp_rejected is not None:
            disp_rejected = disp_rejected[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel_glo = disp_glo * res / dt
    vel_nbs = disp_nbs * res / dt
    vel_spl = disp_spl * res / dt

    # Scatter plot vx(t)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot rejected points if provided
    if disp_rejected is not None:
        vel_rejected = disp_rejected * res / dt
        # Plot all candidate peaks for vx and vy
        ax.plot(np.tile(time[:, None] * 1000, (1, vel_rejected.shape[-2])).flatten(),
                vel_rejected[:, 0, 0, :, 1].flatten(), 'x', c='red', alpha=0.3, ms=2, 
                label='vx (all candidate peaks)')
        ax.plot(np.tile(time[:, None] * 1000, (1, vel_rejected.shape[-2])).flatten(),
                vel_rejected[:, 0, 0, :, 0].flatten(), 'x', c='black', alpha=0.3, ms=2, 
                label='vy (all candidate peaks)')

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

    ax.legend(loc='upper right')
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

    # Calculate statistics with warning suppression for all-NaN slices
    with np.errstate(invalid='ignore'):
        # Plot vy (vertical velocity)
        med_vy = np.nanmedian(vel[:, :, :, 0], axis=(1, 2))
        min_vy = np.nanmin(vel[:, :, :, 0], axis=(1, 2))
        max_vy = np.nanmax(vel[:, :, :, 0], axis=(1, 2))
        
        # Plot vx (horizontal velocity)
        med_vx = np.nanmedian(vel[:, :, :, 1], axis=(1, 2))
        min_vx = np.nanmin(vel[:, :, :, 1], axis=(1, 2))
        max_vx = np.nanmax(vel[:, :, :, 1], axis=(1, 2))

    # Plot vy (vertical velocity)
    ax.plot(time * 1000, med_vy, label='Median vy')

    ax.fill_between(time * 1000, min_vy, max_vy, alpha=0.3, label='Min/max vy')

    # Plot vx (horizontal velocity)
    ax.plot(time * 1000, med_vx, label='Median vx')
    ax.fill_between(time * 1000, min_vx, max_vx, alpha=0.3, label='Min/max vx')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Median velocity in time')
    ax.set(**kwargs)

    ax.legend(loc='upper right')
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax

def plot_vel_Gupta(disp, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.
    



    # Define a time array
    time = get_time(frs, dt)
    #Gupta PLOTTER, Abe
    Flowrate_Gupta,time_Gupta = Gupta_plotter("Male",70,1.90)
    A_coughmachine = 2e-4 #m^2
    
    v_Gupta = Flowrate_Gupta / A_coughmachine / 1000 # v (m/s) = Q (L/s) / A(m), divide Q by a 1000
    ####
    

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp.shape[0] != time.shape[0]:
        disp = disp[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel = disp * res / dt

    # Plot the median velocity in time, show the min and max as a shaded area
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics with warning suppression for all-NaN slices
    with np.errstate(invalid='ignore'):
        # Plot vy (vertical velocity)
        med_vy = np.nanmedian(vel[:, :, :, 0], axis=(1, 2))
        min_vy = np.nanmin(vel[:, :, :, 0], axis=(1, 2))
        max_vy = np.nanmax(vel[:, :, :, 0], axis=(1, 2))
        
        # Plot vx (horizontal velocity)
        med_vx = np.nanmedian(vel[:, :, :, 1], axis=(1, 2))
        min_vx = np.nanmin(vel[:, :, :, 1], axis=(1, 2))
        max_vx = np.nanmax(vel[:, :, :, 1], axis=(1, 2))

    # Plot vy (vertical velocity)
    ax.plot(time * 1000, med_vy, label='Median vy')
    ax.plot(time_Gupta*1000,v_Gupta,label="Gupta",c='k')
    ax.fill_between(time * 1000, min_vy, max_vy, alpha=0.3, label='Min/max vy')

    # Plot vx (horizontal velocity)
    ax.plot(time * 1000, med_vx, label='Median vx')
    ax.fill_between(time * 1000, min_vx, max_vx, alpha=0.3, label='Min/max vx')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Median velocity in time')
    ax.set(**kwargs)

    ax.legend()
    ax.grid()
    plt.show()

    # if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        #save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_prof(disp, res, frs, dt, win_pos, 
                  mode="random", proc_path=None, file_name=None, subfolder=None, test_mode=False, 
                  disp_rejected=None, frame_skip=1, plot_rejected=True, **kwargs):
    
    # Define a time array
    n_corrs = disp.shape[0]
    time = get_time(frs, dt)
    
    # Convert displacement to velocity
    vel = disp * res / dt
    
    # Handle rejected data if provided
    vel_rejected = None
    if disp_rejected is not None:
        vel_rejected = disp_rejected * res / dt
  
    # Raise error if one tries to make a video, but proc_path is not specified
    if mode == "video":
        if test_mode:
            return
        elif proc_path is None or file_name is None:
            raise ValueError("proc_path and file_name must be specified to create a video.")

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
        if mode == "video" and frame_skip > 1:
            # Skip frames: take every frame_skip-th frame
            frames_to_plot = range(frame_skip-1, n_corrs, frame_skip)
        else:
            frames_to_plot = range(n_corrs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'video', 'all', or 'random'.")
    
    # Set up video writer if needed
    if mode == "video":
        fig_video, ax_video = plt.subplots(figsize=(10, 6))
        plt.tight_layout()  # Minimize borders for video frames
        writer = ani.FFMpegWriter(fps=10)
        video_path = os.path.join(proc_path, file_name+'.mp4')
        video_context = writer.saving(fig_video, video_path, dpi=150)
        frames_iter = trange(len(frames_to_plot), desc='Rendering video     ')
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
        
        # Plot rejected points if provided and enabled
        if plot_rejected and vel_rejected is not None:
            vx_rejected = vel_rejected[frame_idx, :, :, :, 1]  # All peaks for vx
            vy_rejected = vel_rejected[frame_idx, :, :, :, 0]  # All peaks for vy
            
            # Create y positions for each peak (repeat y_pos for each peak)
            n_peaks = vel_rejected.shape[-2]
            y_pos_expanded = np.repeat(y_pos[:, np.newaxis], n_peaks, axis=1)
            
            # Plot all rejected peaks as smaller, transparent points
            ax.scatter(vx_rejected.flatten(), y_pos_expanded.flatten(), 
                      c='black', s=10, alpha=0.5, marker='x', label='Rejected vx')
            ax.scatter(vy_rejected.flatten(), y_pos_expanded.flatten(), 
                      c='red', s=10, alpha=0.5, marker='x', label='Rejected vy')
        
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('y position (mm)')
        ax.set_title(f'Velocity profiles at frame {frame_idx + 1} ({time[frame_idx] * 1000:.0f} ms)')
        ax.legend(loc='upper right')
        ax.grid()
        ax.set(**kwargs)
        
        # Minimize white borders
        plt.tight_layout()
    
    # Process frames
    if video_context is not None:
        # Video mode
        with video_context:
            for i in frames_iter:
                frame_idx = frames_to_plot[i]  # Get the actual frame index
                ax_video.clear()
                plot_frame(frame_idx, ax_video)
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


def plot_flow_rate(q, frs, dt, q_model=None, t_model=None, proc_path=None, file_name=None, test_mode=False, 
                   frame_skip=1, plot_model=True, **kwargs):
        
    # Define a time array
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if q.shape[0] != time.shape[0]:
        q = q[frs[0]:frs[-1]]
    
    # Apply frame skipping if specified
    if frame_skip > 1:
        skip_indices = slice(0, len(time), frame_skip)
        time = time[skip_indices]
        q = q[skip_indices]

    # Plot the flow rate in time
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # If a model is provided and plotting is enabled, plot it
    if plot_model and q_model is not None and t_model is not None:
        ax.plot(t_model * 1000, q_model, label='Gupta et al., 2009', c=cvd.get_color(2))

    ax.plot(time * 1000, q * 1000, label='Flow rate')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Flow rate (L/s)')
    ax.set_title('Flow rate in time')
    ax.set(**kwargs)

    if plot_model:
        ax.legend(loc='upper right')

    ax.grid()
    
    # Minimize white borders
    plt.tight_layout()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax