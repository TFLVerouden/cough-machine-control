import re
import xml.etree.ElementTree as ET
import numpy as np
import os


def recursive_search(d, target):
    """
    Recursively search for a key in a nested dictionary.
    
    Parameters
    ----------
    d : dict
        Dictionary to search in
    target : str
        Key to search for
        
    Returns
    -------
    any
        Value of the found key, or None if not found
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if k == target:
                return v
            res = recursive_search(v, target)
            if res is not None:
                return res
    return None


def extract_cihx_metadata(filepath, output_file="cihx_metadata", save=True, 
                          verbose=True):
    """
    Extracts the embedded XML metadata from a .cihx file, prints key info, and saves as a .npz file.
    
    This function extracts ALL metadata keys from the XML and saves them for comprehensive access,
    but only prints the important/commonly used settings to the console for readability.
    
    Parameters
    ----------
    filepath : str
        Path to the .cihx file
    output_file : str
        Base name for the output file (extension will be added automatically)
    save : bool
        Whether to save the metadata to file
    verbose : bool
        Whether to print important extracted metadata to console
        
    Returns
    -------
    dict
        Dictionary containing the extracted metadata
    """
    # Check if input file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find input file: {filepath}")
    
    with open(filepath, "rb") as f:
        data = f.read()
    
    # Extract XML portion
    xml_match = re.search(rb"<cih>.*</cih>", data, flags=re.DOTALL)
    if not xml_match:
        raise ValueError("No XML metadata found in file.")
    
    xml_data = xml_match.group(0).decode("utf-8", errors="ignore")
    root = ET.fromstring(xml_data)
    
    # Convert XML into nested dictionary
    def xml_to_dict(element):
        d = {element.tag: {} if element.attrib else None}
        children = list(element)
        if children:
            dd = {}
            for dc in map(xml_to_dict, children):
                for k, v in dc.items():
                    if k in dd:
                        if not isinstance(dd[k], list):
                            dd[k] = [dd[k]]
                        dd[k].append(v)
                    else:
                        dd[k] = v
            d = {element.tag: dd}
        if element.text and element.text.strip():
            text = element.text.strip()
            if children or element.attrib:
                if text:
                    d[element.tag]["text"] = text
            else:
                d[element.tag] = text
        return d
    
    metadata_dict = xml_to_dict(root)
    
    # Print important camera/recording settings if present
    important_keys = [
        # Camera/system info
        ("date", "Recording date"),
        ("time", "Recording time"),
        ("deviceName", "Camera model"),
        ("firmware", "Firmware version"),

        # Recording parameters
        ("recordRate", "Frame rate [fps]"),
        ("shutterSpeedNsec", "Shutter speed [ns]"),
        ("totalFrame", "Nr of frames"),
        
        # Image properties  
        ("resolution", "Resolution"),
        ("effectiveBit", "Effective bit depth"),
        ("fileFormat", "File format"),
        
        # Image orientation
        ("flipH", "Horizontal flip"),
        ("flipV", "Vertical flip"),
        ("rotate", "Rotation [deg]"),
    ]
    
    print("\n=== Extracted Metadata ===")
    if verbose:
        for key, label in important_keys:
            value = recursive_search(metadata_dict, key)
            if value is not None:
                print(f"{label}: {value}")
    
    # Extract ALL possible keys from the metadata
    def extract_all_keys(d, prefix=""):
        """Extract all keys and their values from nested dictionary"""
        all_keys = {}
        if isinstance(d, dict):
            for k, v in d.items():
                current_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    all_keys.update(extract_all_keys(v, current_key))
                elif isinstance(v, list):
                    all_keys[current_key] = v
                    # Also extract from list items if they're dictionaries
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            all_keys.update(extract_all_keys(item, f"{current_key}[{i}]"))
                else:
                    all_keys[current_key] = v
        return all_keys
    
    # Get all keys and values
    all_metadata_keys = extract_all_keys(metadata_dict)
    
    # Save comprehensive metadata as .npz file
    if save:
        # Ensure output file has correct extension
        if not output_file.endswith('.npz'):
            output_file += '.npz'
        
        # Get directory from input filepath for saving in same location
        input_dir = os.path.dirname(filepath)
        if input_dir:
            output_path = os.path.join(input_dir, output_file)
        else:
            output_path = output_file
        
        # Create save dictionary with source info and original structure
        save_dict = {
            # Source file information
            'source_file_path': filepath,
            'source_file_name': os.path.basename(filepath),
            # Raw metadata (preserves original structure)
            'metadata_dict': metadata_dict,
            # All keys in flattened format
            'all_keys': all_metadata_keys,
        }
        
        # Add each individual key as a separate variable for easy VS Code preview
        # Clean up key names for valid Python variable names
        for key, value in all_metadata_keys.items():
            # Replace dots and brackets with underscores for valid variable names
            clean_key = key.replace('.', '_').replace('[', '_').replace(']', '')
            # Ensure key doesn't start with a number
            if clean_key[0].isdigit():
                clean_key = f"key_{clean_key}"
            save_dict[clean_key] = value
        
        # Save all data
        np.savez(output_path, **save_dict)
        
        print(f"\nMetadata saved to {output_path}")
        print(f"Total keys extracted and saved: {len(all_metadata_keys)}")
        print(f"Individual variables saved: {len(save_dict)}")
    
    return metadata_dict


def list_all_metadata_keys(metadata_dict, prefix=""):
    """
    Recursively prints all metadata keys found in the dictionary with their values.
    
    Parameters
    ----------
    metadata_dict : dict
        Nested dictionary of metadata (from extract_cihx_metadata)
    prefix : str
        Used internally for recursion (for nested keys)
    """
    if isinstance(metadata_dict, dict):
        for k, v in metadata_dict.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                list_all_metadata_keys(v, new_prefix)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    list_all_metadata_keys(item, f"{new_prefix}[{i}]")
            else:
                print(f"{new_prefix}: {v}")
    else:
        print(f"{prefix}: {metadata_dict}")


def load_cihx_metadata(filepath, var_names=None):
    """
    Load previously saved CIHX metadata from an .npz file.
    
    Parameters
    ----------
    filepath : str
        Path to the .npz file containing saved metadata
    var_names : list, optional
        List of variable names to load. If None, loads all variables.
        
    Returns
    -------
    dict
        Dictionary containing the loaded metadata
    """
    # Ensure file has correct extension
    if not filepath.endswith('.npz'):
        filepath += '.npz'
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    # Load the data
    loaded_data = {}
    with np.load(filepath, allow_pickle=True) as data:
        if var_names is None:
            # Load all variables
            for key in data.files:
                loaded_data[key] = data[key].item() if data[key].shape == () else data[key]
        else:
            # Load only requested variables
            for key in var_names:
                if key in data:
                    loaded_data[key] = data[key].item() if data[key].shape == () else data[key]
                else:
                    print(f"Warning: {key} not found in {filepath}")
    
    print(f"Loaded metadata from {filepath}")
    return loaded_data


if __name__ == "__main__":
    print("CIHX metadata extraction tool")
    print("=" * 40)
    print("Select a .cihx file to extract metadata...")
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        
        # Get current directory for initial directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Open file dialogue
        selected_file = filedialog.askopenfilename(
            title="Select a .cihx file",
            filetypes=[("CIHX files", "*.cihx"), ("All files", "*.*")],
            initialdir=current_dir
        )
        
        if selected_file:
            print(f"Selected file: {selected_file}")
            
            # Extract metadata from selected file
            output_name = os.path.splitext(os.path.basename(selected_file))[0] + "_metadata"
            metadata = extract_cihx_metadata(selected_file, output_name, save=True, verbose=True)
        else:
            print("No file selected. Exiting.")
            
    except ImportError:
        print("Error: tkinter not available. Please install tkinter or run this script with a specific file path.")
        print("Usage from code: extract_cihx_metadata('/path/to/file.cihx')")