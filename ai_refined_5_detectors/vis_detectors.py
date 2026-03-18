import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detectors():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    phase_mask_size = config.get('phase_mask_size', [1200, 1200])
    img_size = config.get('img_size', [1000, 1000])
    detector_pos = config.get('detector_pos', [])
    det_size = config.get('detector_size', 120)
    det_shape = config.get('detector_shape', 'square')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Phase Mask boundary
    ax.add_patch(patches.Rectangle((0, 0), phase_mask_size[1], phase_mask_size[0], 
                                  linewidth=2, edgecolor='black', facecolor='none', label='Phase Mask (1200x1200)'))
                                  
    # Plot Image input region (centered)
    pad_x = (phase_mask_size[1] - img_size[1]) / 2
    pad_y = (phase_mask_size[0] - img_size[0]) / 2
    ax.add_patch(patches.Rectangle((pad_x, pad_y), img_size[1], img_size[0], 
                                  linewidth=1, edgecolor='blue', linestyle='--', facecolor='none', alpha=0.5, label='Input Image Region (1000x1000)'))
                                  
    # Plot Detectors
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    if not detector_pos:
        print("Warning: No detector_pos found in config.json")
    
    for i, pos in enumerate(detector_pos):
        # Config pos are [x, y] which often correspond to [row, col] in tensor logic
        # For plotting, x-axis is horizontal (col), y-axis is vertical (row)
        # Assuming pos = [row, col] based on tensor notation used in train.py
        y_center, x_center = pos[0], pos[1] 
        color = colors[i % len(colors)]
        
        if det_shape == 'square':
            # Rectangle needs bottom-left corner
            bottom_left_x = x_center - det_size / 2
            bottom_left_y = y_center - det_size / 2
            rect = patches.Rectangle((bottom_left_x, bottom_left_y), det_size, det_size, 
                                     linewidth=2, edgecolor=color, facecolor=color, alpha=0.4, 
                                     label=f'Det {i}')
            ax.add_patch(rect)
        else:
            # Circle needs center
            circle = patches.Circle((x_center, y_center), det_size / 2, 
                                    linewidth=2, edgecolor=color, facecolor=color, alpha=0.4, 
                                    label=f'Det {i}')
            ax.add_patch(circle)
            
        # Add text label in the center
        ax.text(x_center, y_center, str(i), color='white', weight='bold', 
                fontsize=12, ha='center', va='center')
                
    # Setup axes
    ax.set_xlim(0, phase_mask_size[1])
    ax.set_ylim(phase_mask_size[0], 0) # Invert y-axis to match image/matrix coordinates
    ax.set_aspect('equal')
    ax.set_title(f'Detector Layout Visualization\n(Shape: {det_shape}, Size: {det_size})')
    ax.set_xlabel('X (Columns)')
    ax.set_ylabel('Y (Rows)')
    
    # Place legend outside
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(os.path.dirname(__file__), 'detector_layout_vis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
if __name__ == "__main__":
    visualize_detectors()