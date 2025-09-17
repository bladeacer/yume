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
        venv_uv_path = os.path.join(venv_bin_path, "uv.exe")
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

# --- 4. Core Function: Image Analysis ---
def _analyze_image(image_path, num_colors, images_dir):
    """
    Analyzes a single image and writes results to a markdown file.
    """
    try:
        # Note: These imports are placed inside the function for `multiprocessing` compatibility,
        # as each new process starts with a fresh global namespace.
        from PIL import Image
        from collections import Counter
    except ImportError:
        logger.error("Error: 'Pillow' library not found. Please run 'python setup.py --setup' to install dependencies.")
        # We can't exit the whole program from a child process, so we just return
        return

    # Build the full path to the image using the images_dir
    full_image_path = os.path.join(images_dir, image_path)

    if not os.path.exists(full_image_path):
        logger.error(f"Error: Image file not found at '{full_image_path}'.")
        return

    try:
        image = Image.open(full_image_path)
        # Convert to RGB to ensure consistent 3-tuple representation
        image = image.convert('RGB')
        
        # Get image dimensions and total pixel count
        width, height = image.size
        total_pixels = width * height
        
        # Get file modification timestamp
        mod_timestamp = os.path.getmtime(full_image_path)
        last_modified = datetime.fromtimestamp(mod_timestamp).strftime('%Y-%m-%d %H:%M:%S')

        pixels = list(image.getdata())
        color_counts = Counter(pixels)
        most_common_colors = color_counts.most_common(num_colors)
    except Exception as e:
        logger.error(f"Error processing image '{full_image_path}': {e}")
        return

    # Create the results directory relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate the Markdown output file
    image_file_name = os.path.basename(full_image_path)
    output_filename = os.path.join(results_dir, f"colors_of_{os.path.splitext(image_file_name)[0]}.md")
    
    with open(output_filename, 'w') as f:
        f.write(f"# Color Analysis of `{image_file_name}`\n\n")
        
        # Add new image details section
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
            
    logger.info(f"Successfully wrote color analysis results to '{output_filename}'.")

# --- 5. Core Function: Create Summary ---
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
    
    logger.info("Starting summary creation. Scanning for analysis files...")
    
    for filename in os.listdir(results_dir):
        if filename.startswith("colors_of_") and filename.endswith(".md"):
            file_path = os.path.join(results_dir, filename)
            
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Extract image details using a more robust parsing method
                details = {}
                lines = content.splitlines()
                
                in_details_section = False
                for line in lines:
                    if "## Image Details" in line:
                        in_details_section = True
                        continue # Move to the next line to start parsing
                    
                    # Stop parsing details when a new section starts
                    if "## Most Frequent Colors" in line:
                        break
                        
                    if in_details_section:
                        if "**Dimensions:**" in line:
                            details['dimensions'] = line.split('`')[1] if '`' in line else 'N/A'
                        elif "**Total Pixels:**" in line:
                            details['pixels'] = line.split('`')[1] if '`' in line else 'N/A'
                        elif "**Last Modified:**" in line:
                            details['modified'] = line.split('`')[1] if '`' in line else 'N/A'
                            
                # Append the dictionary containing both filename and details
                analyzed_files.append({
                    'filename': filename,
                    'details': details
                })
                
                # Extract color counts with more robust parsing
                for line in content.splitlines():
                    if line.strip().startswith('| `('):
                        parts = [p.strip() for p in line.split('|')]
                        try:
                            # Extract the string inside the parentheses
                            rgb_str = parts[1].strip('` ').strip('()')
                            # Split by comma and convert to integers
                            r, g, b = [int(val.strip()) for val in rgb_str.split(',')]
                            color_tuple = (r, g, b)
                            count = int(parts[3].strip())
                            all_color_counts[color_tuple] += count
                        except (ValueError, IndexError, TypeError) as e:
                            logger.warning(f"Could not parse color data from line in {filename}: {line.strip()}. Error: {e}")
    
    # Filter out files without details from the list
    files_with_details = [f for f in analyzed_files if f.get('details') and f['details'].get('dimensions')]

    # Find the top 10 most frequent colors overall
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

# --- 6. Main Script Logic ---
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
    # This is required for multiprocessing to work on all platforms
    main()

