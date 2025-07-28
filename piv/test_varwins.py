import numpy as np
import matplotlib.pyplot as plt

def split_n_shift_4d(img: np.ndarray, n_wins: tuple[int, int], overlap: float = 0, 
                     shift: tuple[int, int] | np.ndarray = (0, 0), 
                     shift_mode: str = 'before', plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 2D image array (y, x) into (overlapping) windows,
    with automatic window size adjustments for shifted images.
    Now supports per-window shifts via 4D shift arrays.

    Args:
        img (np.ndarray): 2D array of image values (y, x).
        n_wins (tuple[int, int]): Number of windows in (y, x) direction.
        overlap (float): Fractional overlap between windows (0 = no overlap).
        shift (tuple[int, int] | np.ndarray): 
            - (dy, dx) shift in pixels for uniform shift
            - 2D array (n_y, n_x, 2) for per-window shifts
        shift_mode (str): 'before' or 'after' shift: which frame is considered?
        plot (bool): If True, plot the windows on the image.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - wins: 4D array of image windows (window_y_idx, window_x_idx, y, x)
            - win_pos: corresponding positions (window_y_idx, window_x_idx, 2)
    """
    # Get dimensions
    h, w = img.shape
    n_y, n_x = n_wins
    
    # Handle different shift formats
    if isinstance(shift, (tuple, list)) or (isinstance(shift, np.ndarray) and shift.ndim == 1):
        # Uniform shift - convert to per-window format
        dy, dx = np.asarray(shift, dtype=int)
        shifts_per_window = np.full((n_y, n_x, 2), [dy, dx], dtype=int)
    elif isinstance(shift, np.ndarray) and shift.shape == (n_y, n_x, 2):
        # Per-window shifts
        shifts_per_window = shift.astype(int)
    else:
        raise ValueError(f"Invalid shift format. Expected tuple, 1D array, or array of shape ({n_y}, {n_x}, 2)")
    
    # Find the maximum absolute shift to determine window size cutoff
    max_dy = np.max(np.abs(shifts_per_window[:, :, 0]))
    max_dx = np.max(np.abs(shifts_per_window[:, :, 1]))
    
    # Warn if shifts are very large
    if max_dy > h // 4 or max_dx > w // 4:
        print(f"Warning: Large shifts detected (max_dy={max_dy}, max_dx={max_dx}). "
              f"This may result in very small windows.")
    
    # Calculate window size including overlap
    size_y = min(int(h // n_y * (1 + overlap)), h)
    size_x = min(int(w // n_x * (1 + overlap)), w)
    
    # Get the top-left corner of each window to create grid of window coords
    y_indices = np.linspace(0, h - size_y, num=n_y, dtype=int)
    x_indices = np.linspace(0, w - size_x, num=n_x, dtype=int)
    grid = np.stack(np.meshgrid(y_indices, x_indices, indexing="ij"), axis=-1)

    # Compute physical centres of windows in image coordinates (for plotting/visualization)
    centres = np.stack((grid[:, :, 0] + size_y / 2,
                        grid[:, :, 1] + size_x / 2), axis=-1)

    # Determine cut-off direction: +1 for 'before', -1 for 'after'
    mode_sign = 1 if shift_mode == 'after' else -1

    # Show windows and centres on the image if requested
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img.astype(float) / img.max() * 255, cmap='gray')

    # Pre-allocate windows with size based on maximum shifts
    windows = np.empty((n_y, n_x, size_y - max_dy, size_x - max_dx), dtype=img.dtype)
    
    for i, y in enumerate(y_indices):
        for j, x in enumerate(x_indices):
            # Get the specific shift for this window
            dy, dx = shifts_per_window[i, j]
            
            # Calculate cut-off for this specific window
            cut_y0 = max(0, mode_sign * dy)
            cut_y1 = max(0, -mode_sign * dy)
            cut_x0 = max(0, mode_sign * dx)
            cut_x1 = max(0, -mode_sign * dx)
            
            # Extract window with this window's specific shift
            y0 = y + cut_y0
            y1 = y + size_y - cut_y1
            x0 = x + cut_x0
            x1 = x + size_x - cut_x1
            
            # Get the window and pad if necessary to maintain consistent output size
            window = img[y0:y1, x0:x1]
            
            # Pad to match the maximum shift size if this window's shift is smaller
            pad_y = (max_dy - abs(dy)) // 2
            pad_x = (max_dx - abs(dx)) // 2
            if pad_y > 0 or pad_x > 0:
                window = np.pad(window, ((pad_y, max_dy - abs(dy) - pad_y), 
                                       (pad_x, max_dx - abs(dx) - pad_x)), 
                              mode='edge')
            
            windows[i, j] = window

            if plot:
                color = ['orange', 'blue', 'red', 'green'][(i + j) % 4]
                rect = plt.Rectangle((x + cut_x0, y + cut_y0),
                                   x1 - x0, y1 - y0,
                                   edgecolor=color, facecolor='none',
                                   linewidth=2)
                ax.add_patch(rect)
                ax.scatter(centres[i, j, 1], centres[i, j, 0], c=color,
                          marker='x', s=60)
                
                # Add shift annotation
                ax.annotate(f'({dy},{dx})', 
                          (centres[i, j, 1], centres[i, j, 0]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, color=color, weight='bold')

    # Finish plot
    if plot:
        ax.set_xlim(-20, w + 20)
        ax.set_ylim(-20, h + 20)
        ax.set_title('Windows with per-window shifts')
        plt.show()

    return windows, centres


# Test the function
def test_split_n_shift_4d():
    # Create a simple test image with some pattern
    img = np.zeros((100, 120))
    
    # Add some features to make shifts visible
    img[20:25, 10:15] = 255  # Top-left bright square
    img[80:85, 100:105] = 255  # Bottom-right bright square
    img[40:60, 50:70] = 128   # Center gray rectangle
    
    # Add diagonal gradient
    for i in range(100):
        for j in range(120):
            img[i, j] += (i + j) * 0.5
    
    n_wins = (3, 2)  # 3 rows, 2 columns
    
    # Test 1: Uniform shifts (should work as before)
    print("Test 1: Uniform shifts")
    shift_uniform = (5, -3)
    windows1, centers1 = split_n_shift_4d(img, n_wins, shift=shift_uniform, plot=True)
    print(f"Windows shape: {windows1.shape}")
    print(f"Centers shape: {centers1.shape}")
    
    # Test 2: Per-window shifts
    print("\nTest 2: Per-window shifts")
    shifts_per_window = np.array([
        [[0, 0], [5, 2]],      # Row 0: no shift, then (5,2)
        [[3, -4], [0, 0]],     # Row 1: (3,-4), then no shift  
        [[-2, 3], [8, -1]]     # Row 2: (-2,3), then (8,-1)
    ])
    
    windows2, centers2 = split_n_shift_4d(img, n_wins, shift=shifts_per_window, plot=True)
    print(f"Windows shape: {windows2.shape}")
    print(f"Centers shape: {centers2.shape}")
    
    # Test 3: Show individual windows
    print("\nTest 3: Individual windows visualization")
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i in range(3):
        for j in range(2):
            ax = axes[i, j]
            ax.imshow(windows2[i, j], cmap='gray')
            shift = shifts_per_window[i, j]
            ax.set_title(f'Window ({i},{j})\nShift: ({shift[0]},{shift[1]})')
            ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    return windows1, centers1, windows2, centers2

if __name__ == "__main__":
    windows1, centers1, windows2, centers2 = test_split_n_shift_4d()