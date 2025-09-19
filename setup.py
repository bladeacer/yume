#!/usr/bin/env python
import argparse
import os
import subprocess
import sys
import shutil
import logging
from PIL import Image
from collections import Counter
import ast
from datetime import datetime
import multiprocessing
import math

# --- 1. Basic Logging Setup ---
# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Helper Functions for Virtual Environment Paths ---
def _get_venv_paths(venv_path):
    """
    Determines the paths to the python, pip, and uv executables
    within the virtual environment, based on the operating system.
    """
    if sys.platform == "win32":
        # On Windows, executables are in 'Scripts'
        venv_bin_path = os.path.join(venv_path, "Scripts")
        venv_python_path = os.path.join(venv_bin_path, "python.exe")
        venv_pip_path = os.path.join(venv_bin_path, "pip.exe")
        venv_uv_path = os.path.join(venv_bin_path, "uv")
    else:
        # On Linux/macOS, executables are in 'bin'
        venv_bin_path = os.path.join(venv_path, "bin")
        venv_python_path = os.path.join(venv_bin_path, "python")
        venv_pip_path = os.path.join(venv_bin_path, "pip")
        venv_uv_path = os.path.join(venv_bin_path, "uv")

    return venv_python_path, venv_pip_path, venv_uv_path

# --- 3. Core Function: Setup Virtual Environment ---
def _setup_venv(venv_path, requirements_file):
    """
    Creates a .venv virtual environment, installs uv, and then installs
    dependencies from requirements_file using uv.
    """
    logger.info(f"Starting virtual environment setup at: {venv_path}")

    # Remove existing venv for a clean setup
    if os.path.exists(venv_path):
        logger.info(f"Removing existing virtual environment at: {venv_path}")
        try:
            shutil.rmtree(venv_path)
            logger.info("Existing virtual environment removed successfully.")
        except OSError as e:
            logger.error(f"Error removing virtual environment '{venv_path}': {e}")
            sys.exit(1)

    # Create a new virtual environment
    logger.info(f"Creating new virtual environment at: {venv_path}")
    try:
        # Use the system's default Python to create the venv
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
        logger.info("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during venv creation: {e}")
        sys.exit(1)

    # Get paths to executables within the new venv
    venv_python_path, venv_pip_path, venv_uv_path = _get_venv_paths(venv_path)

    # Ensure pip is up-to-date in the venv (optional but good practice)
    logger.info("Upgrading pip in the virtual environment...")
    try:
        subprocess.run([venv_python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
        logger.info("pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upgrade pip: {e}") # Warning, not critical error

    # Install uv using pip within the venv
    logger.info(f"Installing 'uv' into '{venv_path}'...")
    try:
        subprocess.run([venv_pip_path, "install", "uv"], check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
        logger.info("'uv' installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install 'uv'. Command: {' '.join(e.cmd)} Error: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error(f"Error: pip executable not found at '{venv_pip_path}'. Make sure the venv is correctly set up.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while installing 'uv': {e}")
        sys.exit(1)

    # Install Python dependencies from requirements.txt using uv
    if os.path.exists(requirements_file):
        logger.info(f"Installing Python dependencies from '{requirements_file}' using 'uv'...")
        try:
            subprocess.run([venv_uv_path, "pip", "install", "-r", requirements_file], check=True,
                           stdout=sys.stdout, stderr=sys.stderr)
            logger.info(f"Python dependencies installed successfully into '{venv_path}' using 'uv'.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Python dependencies using 'uv'. Command: {' '.join(e.cmd)} Error: {e}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"Error: uv executable not found at '{venv_uv_path}'. 'uv' might not have installed correctly.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"An unexpected error occurred while installing Python dependencies with 'uv': {e}")
            sys.exit(1)
    else:
        logger.warning(f"Requirements file '{requirements_file}' not found. Skipping dependency installation.")

    logger.info("Virtual environment setup complete.")

# --- 4. Helper Functions for Color Analysis ---
def _calculate_color_distance(color1, color2):
    """Calculates the Euclidean distance between two RGB colors."""
    return math.sqrt(
        (color1[0] - color2[0])**2 +
        (color1[1] - color2[1])**2 +
        (color1[2] - color2[2])**2
    )

def _calculate_mixed_color(colors_in_group):
    """Calculates the average color of a group of RGB colors."""
    if not colors_in_group:
        return (0, 0, 0)
    
    total_r = sum(c[0] for c in colors_in_group)
    total_g = sum(c[1] for c in colors_in_group)
    total_b = sum(c[2] for c in colors_in_group)
    count = len(colors_in_group)
    
    return (int(total_r / count), int(total_g / count), int(total_b / count))

# A set of common colors to group by
TARGET_COLORS = {
    "Red": (255, 0, 0),
    "Green": (0, 255, 0),
    "Blue": (92, 1, 254),
    "Yellow": (243, 225, 107),
    "Orange": (255, 165, 0),
    "Pink": (255, 192, 203),
    "Cyan": (98, 124, 153),
    "Purple": (255, 0, 255),
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Dark Blue": (36, 46, 68),
    "Earthy Green": (66, 67, 61),
    "Gray": (44, 50, 57),
    "Dark Purple": (30, 55, 151),
    "Gray Blue": (51, 75, 105)
}


# --- 5. Core Function: Image Analysis ---
def _analyze_image(image_path, num_colors, images_dir):
    """
    Analyzes a single image and writes results to a markdown file.
    """
    try:
        from PIL import Image
        from collections import Counter
    except ImportError:
        logger.error("Error: 'Pillow' library not found. Please run 'python setup.py --setup' to install dependencies.")
        return

    full_image_path = os.path.join(images_dir, image_path)

    if not os.path.exists(full_image_path):
        logger.error(f"Error: Image file not found at '{full_image_path}'.")
        return

    try:
        image = Image.open(full_image_path)
        image = image.convert('RGB')
        
        width, height = image.size
        total_pixels = width * height
        
        mod_timestamp = os.path.getmtime(full_image_path)
        last_modified = datetime.fromtimestamp(mod_timestamp).strftime('%Y-%m-%d %H:%M:%S')

        pixels = list(image.getdata())
        color_counts = Counter(pixels)
        most_common_colors = color_counts.most_common(num_colors)
    except Exception as e:
        logger.error(f"Error processing image '{full_image_path}': {e}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    image_file_name = os.path.basename(full_image_path)
    output_filename = os.path.join(results_dir, f"colors_of_{os.path.splitext(image_file_name)[0]}.md")
    
    # --- New Color Grouping Logic ---
    color_groups = {color_name: [] for color_name in TARGET_COLORS}
    
    for color, count in most_common_colors:
        best_match_name = None
        min_distance = float('inf')
        
        for target_name, target_color in TARGET_COLORS.items():
            distance = _calculate_color_distance(color, target_color)
            if distance < min_distance:
                min_distance = distance
                best_match_name = target_name
        
        if best_match_name:
            color_groups[best_match_name].append({
                "color": color,
                "count": count,
                "distance": min_distance
            })

    with open(output_filename, 'w') as f:
        f.write(f"# Color Analysis of `{image_file_name}`\n\n")
        
        f.write("## Image Details\n\n")
        f.write(f"- **Dimensions:** `{width} x {height}`\n")
        f.write(f"- **Total Pixels:** `{total_pixels}`\n")
        f.write(f"- **Last Modified:** `{last_modified}`\n\n")

        f.write("## Most Frequent Colors\n\n")
        f.write(f"This analysis found the top **{len(most_common_colors)}** most frequent colors in the image.\n\n")
        f.write("| Color (RGB) | Color (Hex) | Count | Color Swatch |\n")
        f.write("| :--- | :--- | :--- | :---: |\n")
        
        for color, count in most_common_colors:
            r, g, b = color
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            f.write(f"| `{color}` | `{hex_color}` | {count} | <span style='background-color:{hex_color}; display:inline-block; width:20px; height:20px; border:1px solid #ccc;'></span> |\n")

        f.write("\n---\n\n")
        f.write("## Color Group Analysis\n\n")
        f.write("This section groups the most frequent colors by their similarity to a set of predefined colors.\n")
        f.write("The 'Mixed Color' is the average RGB of all colors in the group.\n")

        for group_name, colors_in_group in color_groups.items():
            if not colors_in_group:
                continue

            mixed_color = _calculate_mixed_color([c['color'] for c in colors_in_group])
            r, g, b = mixed_color
            mixed_hex = f"#{r:02x}{g:02x}{b:02x}"

            f.write(f"\n### Colors Resembling {group_name}\n\n")
            f.write(f"**Mixed Color:** `{mixed_color}` (`{mixed_hex}`) ")
            f.write(f"<span style='background-color:{mixed_hex}; display:inline-block; width:20px; height:20px; border:1px solid #ccc;'></span>\n\n")
            
            # Sort colors by distance (lower is better) and take top 5
            sorted_colors = sorted(colors_in_group, key=lambda x: x['distance'])
            top_x_similar = sorted_colors[:10]

            f.write("| Color (RGB) | Color (Hex) | Count | Similarity Score | Color Swatch |\n")
            f.write("| :--- | :--- | :--- | :--- | :---: |\n")

            for item in top_x_similar:
                color, count, distance = item['color'], item['count'], item['distance']
                r, g, b = color
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                f.write(f"| `{color}` | `{hex_color}` | {count} | `{distance:.2f}` | <span style='background-color:{hex_color}; display:inline-block; width:20px; height:20px; border:1px solid #ccc;'></span> |\n")
            
    logger.info(f"Successfully wrote color analysis results to '{output_filename}'.")


# --- 6. Core Function: Create Summary ---
def _create_summary():
    """
    Scans all image analysis markdown files and creates a summary.md file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    summary_file = os.path.join(results_dir, "summary.md")
    
    if not os.path.exists(results_dir):
        logger.warning(f"Results directory '{results_dir}' not found. Cannot create summary.")
        return
        
    all_color_counts = Counter()
    analyzed_files = []
    
    # New data structure to aggregate unique, grouped colors from all files
    all_grouped_colors = {group_name: {} for group_name in TARGET_COLORS.keys()}

    logger.info("Starting summary creation. Scanning for analysis files...")
    
    for filename in os.listdir(results_dir):
        if filename.startswith("colors_of_") and filename.endswith(".md"):
            file_path = os.path.join(results_dir, filename)
            
            with open(file_path, 'r') as f:
                content = f.read()
                
                details = {}
                lines = content.splitlines()
                
                in_details_section = False
                in_grouped_colors_section = False
                current_group_name = None

                for line in lines:
                    if "## Image Details" in line:
                        in_details_section = True
                        continue
                    
                    if "## Most Frequent Colors" in line:
                        in_details_section = False
                        in_grouped_colors_section = False
                        continue
                    
                    if "## Color Group Analysis" in line:
                        in_grouped_colors_section = True
                        continue
                    
                    # Parse image details section
                    if in_details_section:
                        if "**Dimensions:**" in line:
                            details['dimensions'] = line.split('`')[1] if '`' in line else 'N/A'
                        elif "**Total Pixels:**" in line:
                            details['pixels'] = line.split('`')[1] if '`' in line else 'N/A'
                        elif "**Last Modified:**" in line:
                            details['modified'] = line.split('`')[1] if '`' in line else 'N/A'

                    # Parse color group tables
                    if in_grouped_colors_section:
                        if line.startswith('### Colors Resembling '):
                            current_group_name = line.replace('### Colors Resembling ', '').strip()
                            continue
                        if current_group_name and line.strip().startswith('| `('):
                            parts = [p.strip() for p in line.split('|')]
                            try:
                                rgb_str = parts[1].strip('` ').strip('()')
                                r, g, b = [int(val.strip()) for val in rgb_str.split(',')]
                                color_tuple = (r, g, b)
                                count = int(parts[3].strip())
                                distance = float(parts[4].strip('` '))
                                
                                # Only add colors with a similarity score of 35 or less
                                if distance <= 35:
                                    # Use the color tuple as a key for uniqueness
                                    if color_tuple not in all_grouped_colors[current_group_name] or \
                                       distance < all_grouped_colors[current_group_name][color_tuple]['distance']:
                                        all_grouped_colors[current_group_name][color_tuple] = {
                                            "color": color_tuple,
                                            "count": count,
                                            "distance": distance,
                                        }
                            except (ValueError, IndexError, TypeError) as e:
                                logger.warning(f"Failed to parse line in '{filename}': {line}. Error: {e}")
                                
                analyzed_files.append({
                    'filename': filename,
                    'details': details
                })
                
                # For the top 10 summary, we still use the old method
                for line in content.splitlines():
                    if line.strip().startswith('| `('):
                        parts = [p.strip() for p in line.split('|')]
                        try:
                            rgb_str = parts[1].strip('` ').strip('()')
                            r, g, b = [int(val.strip()) for val in rgb_str.split(',')]
                            color_tuple = (r, g, b)
                            count = int(parts[3].strip())
                            all_color_counts[color_tuple] += count
                        except (ValueError, IndexError, TypeError):
                            pass
    
    files_with_details = [f for f in analyzed_files if f.get('details') and f['details'].get('dimensions')]
    most_common_overall = all_color_counts.most_common(10)
    
    with open(summary_file, 'w') as f:
        f.write("# Overall Image Analysis Summary\n\n")
        f.write("This summary provides a quick overview of the color and technical analysis results for all files in the `results/` directory.\n\n")
        
        f.write("## Top 10 Most Frequent Colors Overall\n\n")
        if most_common_overall:
            f.write("The following colors are the most frequent across all analyzed images.\n\n")
            f.write("| Rank | Color (RGB) | Color (Hex) | Count | Color Swatch |\n")
            f.write("| :--- | :--- | :--- | :--- | :---: |\n")
            for i, (color, count) in enumerate(most_common_overall):
                r, g, b = color
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                f.write(f"| {i+1} | `{color}` | `{hex_color}` | {count} | <span style='background-color:{hex_color}; display:inline-block; width:20px; height:20px; border:1px solid #ccc;'></span> |\n")
        else:
            f.write("No color data was found to summarize.\n")

        f.write("\n---\n\n")
        f.write("## Overall Color Group Analysis\n\n")
        f.write("This section aggregates unique, similar colors from all images into groups. Only colors with a similarity score of 35 or less are included, as a lower score indicates a closer match.\n")
        f.write("The 'Mixed Color' is the average RGB of all unique colors in the group across all images.\n")

        for group_name, colors_in_group_dict in all_grouped_colors.items():
            if not colors_in_group_dict:
                continue

            # Get the list of color data objects
            colors_in_group = list(colors_in_group_dict.values())
            
            mixed_color = _calculate_mixed_color([c['color'] for c in colors_in_group])
            r, g, b = mixed_color
            mixed_hex = f"#{r:02x}{g:02x}{b:02x}"

            f.write(f"\n### All Colors Resembling {group_name}\n\n")
            f.write(f"**Mixed Color:** `{mixed_color}` (`{mixed_hex}`) ")
            f.write(f"<span style='background-color:{mixed_hex}; display:inline-block; width:20px; height:20px; border:1px solid #ccc;'></span>\n\n")
            
            # Sort colors by distance (lower is better) and take top 5
            sorted_colors = sorted(colors_in_group, key=lambda x: x['distance'])
            top_x_similar = sorted_colors[:10]

            f.write("| Color (RGB) | Color (Hex) | Count | Similarity Score | Color Swatch |\n")
            f.write("| :--- | :--- | :--- | :--- | :---: |\n")

            for item in top_x_similar:
                color, count, distance = item['color'], item['count'], item['distance']
                r, g, b = color
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                f.write(f"| `{color}` | `{hex_color}` | {count} | `{distance:.2f}` | <span style='background-color:{hex_color}; display:inline-block; width:20px; height:20px; border:1px solid #ccc;'></span> |\n")

        f.write("\n---\n\n")
        f.write("## Analyzed Image Details\n\n")
        if files_with_details:
            f.write("This table shows the technical details for each analyzed image.\n\n")
            f.write("| Filename | Dimensions | Total Pixels | Last Modified |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for file_data in files_with_details:
                details = file_data['details']
                f.write(f"| `{file_data['filename']}` | {details.get('dimensions', 'N/A')} | {details.get('pixels', 'N/A')} | {details.get('modified', 'N/A')} |\n")
        else:
            f.write("No image analysis files were found with technical details to summarize.\n")

    logger.info(f"Successfully created summary file at '{summary_file}'.")

# --- 7. Main Script Logic ---
def main():
    parser = argparse.ArgumentParser(
        description="Manages project setup and image analysis."
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create/recreate .venv and install dependencies from requirements.txt."
    )
    parser.add_argument(
        "--venv-name",
        default=".venv",
        help="Name of the virtual environment directory (default: .venv)."
    )
    parser.add_argument(
        "--reqs-file",
        default="requirements.txt",
        help="Path to the requirements file (default: requirements.txt)."
    )
    parser.add_argument(
        "--analyze-image",
        type=str,
        nargs='?', # Makes the argument optional
        const='all', # Default value if flag is present without a path
        help="Analyze a single image (e.g., --analyze-image my_photo.jpg) or analyze all images recursively in the images/ directory (e.g., --analyze-image)."
    )
    parser.add_argument(
        "--images-dir",
        default="images",
        help="Path to the directory containing images (default: images)."
    )
    parser.add_argument(
        "--num-colors",
        type=int,
        default=10,
        help="Number of most frequent colors to display (default: 10)."
    )
    parser.add_argument(
        "--create-summary",
        action="store_true",
        help="Create a summary of all image analysis results in the results/ directory."
    )

    args = parser.parse_args()

    # Define absolute paths
    venv_path = os.path.abspath(args.venv_name)
    requirements_file = os.path.abspath(args.reqs_file)

    # If no flags are provided, show help and exit
    if not any([args.setup, args.analyze_image, args.create_summary]):
        parser.print_help()
        sys.exit(0)

    # --- Handle --setup flag ---
    if args.setup:
        _setup_venv(venv_path, requirements_file)
        logger.info("Setup process completed.")

    # --- Handle --analyze-image flag ---
    if args.analyze_image:
        if args.analyze_image == 'all':
            logger.info(f"Analyzing all images recursively in the '{args.images_dir}' directory...")
            
            # Use multiprocessing for parallel analysis
            image_files_to_analyze = []
            for dirpath, dirnames, filenames in os.walk(args.images_dir):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                        relative_path = os.path.relpath(os.path.join(dirpath, filename), args.images_dir)
                        image_files_to_analyze.append((relative_path, args.num_colors, args.images_dir))
            
            if image_files_to_analyze:
                with multiprocessing.Pool() as pool:
                    pool.starmap(_analyze_image, image_files_to_analyze)
                
            _create_summary()
        else:
            _analyze_image(args.analyze_image, args.num_colors, args.images_dir)
            _create_summary()

    # --- Handle --create-summary flag ---
    if args.create_summary:
        _create_summary()

if __name__ == "__main__":
    main()

