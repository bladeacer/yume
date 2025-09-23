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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import threading
import itertools
import sqlite3
import json

# --- 1. Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. Helper Functions for Virtual Environment Paths ---
def _get_venv_paths(venv_path):
    """
    Determines the paths to the python, pip, and uv executables
    within the virtual environment, based on the operating system.
    """
    if sys.platform == "win32":
        venv_bin_path = os.path.join(venv_path, "Scripts")
        venv_python_path = os.path.join(venv_bin_path, "python.exe")
        venv_pip_path = os.path.join(venv_bin_path, "pip.exe")
        venv_uv_path = os.path.join(venv_bin_path, "uv")
    else:
        venv_bin_path = os.path.join(venv_path, "bin")
        venv_python_path = os.path.join(venv_bin_path, "python")
        venv_pip_path = os.path.join(venv_bin_path, "pip")
        venv_uv_path = os.path.join(venv_bin_path, "uv")

    return venv_python_path, venv_pip_path, venv_uv_path

# --- 3. Core Function: Setup Virtual Environment ---
def _setup_venv(venv_path, requirements_file):
    """
    Creates a .venv virtual environment, installs uv, and then installs
    dependencies from requirements.txt using uv.
    """
    logger.info(f"Starting virtual environment setup at: {venv_path}")
    if os.path.exists(venv_path):
        logger.info(f"Removing existing virtual environment at: {venv_path}")
        try:
            shutil.rmtree(venv_path)
            logger.info("Existing virtual environment removed successfully.")
        except OSError as e:
            logger.error(f"Error removing virtual environment '{venv_path}': {e}")
            sys.exit(1)

    logger.info(f"Creating new virtual environment at: {venv_path}")
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
        logger.info("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create virtual environment: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during venv creation: {e}")
        sys.exit(1)

    venv_python_path, venv_pip_path, venv_uv_path = _get_venv_paths(venv_path)

    logger.info("Upgrading pip in the virtual environment...")
    try:
        subprocess.run([venv_python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True,
                       stdout=sys.stdout, stderr=sys.stderr)
        logger.info("pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to upgrade pip: {e}")

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

# --- 5. Core Function: Image Analysis & Clustering ---
def _analyze_image_for_clustering(image_path, images_dir, num_colors):
    """
    Analyzes a single image and returns a list of its most frequent colors.
    """
    try:
        from PIL import Image
        from collections import Counter
    except ImportError:
        logger.error("Error: 'Pillow' library not found.")
        return []

    full_image_path = os.path.join(images_dir, image_path)
    if not os.path.exists(full_image_path):
        logger.error(f"Error: Image file not found at '{full_image_path}'.")
        return []

    try:
        image = Image.open(full_image_path)
        image = image.convert('RGB')
        
        pixels = list(image.getdata())
        color_counts = Counter(pixels)
        most_common_colors_with_counts = color_counts.most_common(num_colors)
        
        logger.info(f"Successfully collected {len(most_common_colors_with_counts)} colors from '{image_path}'.")
        return most_common_colors_with_counts
            
    except Exception as e:
        logger.error(f"Error processing image '{full_image_path}': {e}")
        return []

def _cluster_all_colors(all_color_data, num_clusters):
    """
    Performs k-means clustering on all collected color data.
    Returns: (list of cluster centers, list of labels for each color)
    """
    if not all_color_data:
        return [], []

    flat_colors = [item[0] for item in all_color_data]
    colors_array = np.array(flat_colors, dtype=np.float64)

    if len(colors_array) < num_clusters:
        num_clusters = len(colors_array)
        logger.warning(f"Number of colors is less than clusters. Reducing clusters to {num_clusters}.")
    
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(colors_array)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    return cluster_centers, labels

# --- Persistence Functions ---
def _save_analysis_to_db(db_path, all_colors_data, cluster_centers, labels):
    """Saves the analysis results to a SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS analysis_results")
        cursor.execute("""
            CREATE TABLE analysis_results (
                id INTEGER PRIMARY KEY,
                color TEXT NOT NULL,
                count INTEGER,
                cluster_id INTEGER
            )
        """)
        
        data_to_insert = []
        for i, color_with_count in enumerate(all_colors_data):
            color, count = color_with_count
            color_json = json.dumps(color)
            data_to_insert.append((color_json, count, int(labels[i])))

        cursor.executemany("INSERT INTO analysis_results (color, count, cluster_id) VALUES (?, ?, ?)", data_to_insert)

        cursor.execute("DROP TABLE IF EXISTS cluster_centers")
        cursor.execute("""
            CREATE TABLE cluster_centers (
                cluster_id INTEGER PRIMARY KEY,
                center_rgb TEXT NOT NULL
            )
        """)
        for i, center in enumerate(cluster_centers):
            cursor.execute("INSERT INTO cluster_centers (cluster_id, center_rgb) VALUES (?, ?)", (i, json.dumps(center.tolist())))

        conn.commit()
        logger.info(f"Analysis results saved to '{db_path}' successfully.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error while saving to database: {e}")
    finally:
        if conn:
            conn.close()

def _load_analysis_from_db(db_path):
    """Loads the analysis results from a SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'")
        if cursor.fetchone() is None:
            logger.warning("Database schema not found. Re-running analysis.")
            return None, None, None

        cursor.execute("SELECT color, count, cluster_id FROM analysis_results")
        rows = cursor.fetchall()

        all_colors_data = []
        labels = []
        for color_json, count, cluster_id in rows:
            color = tuple(json.loads(color_json))
            all_colors_data.append((color, count))
            labels.append(cluster_id)

        cursor.execute("SELECT center_rgb FROM cluster_centers ORDER BY cluster_id")
        center_rows = cursor.fetchall()
        cluster_centers = np.array([json.loads(row[0]) for row in center_rows])
        
        logger.info(f"Analysis results loaded from '{db_path}' successfully.")
        return all_colors_data, cluster_centers, np.array(labels)
    except sqlite3.Error as e:
        logger.error(f"SQLite error while loading from database: {e}")
        return None, None, None
    finally:
        if conn:
            conn.close()

# --- Markdown Output Function ---
def _save_to_markdown(all_grouped_colors, output_path="clusters.md"):
    """Saves the color analysis results to a Markdown file."""
    try:
        with open(output_path, "w") as f:
            f.write("# Color Cluster Analysis\n\n")
            f.write("Generated from image analysis.\n\n")
            
            def rgb_to_hex(rgb):
                return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'

            for group_center, group_data in all_grouped_colors.items():
                mixed_color = group_data["mixed_color"]
                mixed_hex = rgb_to_hex(mixed_color)
                
                f.write(f"## Cluster Center: {mixed_color} ({mixed_hex})\n\n")
                
                f.write("| Swatch | RGB | Hex | Count | Similarity Score |\n")
                f.write("| :---: | :---: | :---: | :---: | :---: |\n")
                
                sorted_colors = sorted(group_data["colors"], key=lambda x: x['distance'])[:10]
                
                for item in sorted_colors:
                    color = item['color']
                    count = item['count']
                    distance = item['distance']
                    hex_color = rgb_to_hex(color)
                    
                    swatch_emoji = f'<span style="background-color: {hex_color}; border: 1px solid black; padding: 0.5em 1em; display: inline-block;"></span>'
                    
                    f.write(f"| {swatch_emoji} | {color} | `{hex_color}` | {count} | {distance:.2f} |\n")
                f.write("\n")
                
        logger.info(f"Color analysis results saved to '{output_path}' successfully.")
    except Exception as e:
        logger.error(f"Error writing to Markdown file: {e}")

# --- RGB Graph Plotting Function ---
def _plot_cluster_graph(all_colors_array, cluster_centers, labels, ax):
    """
    Plots a 3D RGB scatter graph showing all colors and their cluster centers.
    """
    if len(all_colors_array) == 0 or len(cluster_centers) == 0:
        ax.text(0.5, 0.5, 0.5, "No data to display", color='red', transform=ax.transAxes)
        return

    colors_by_label = [plt.cm.viridis(label / len(cluster_centers)) for label in labels]
    
    ax.scatter(all_colors_array[:, 0], all_colors_array[:, 1], all_colors_array[:, 2], 
               c=colors_by_label, s=20, alpha=0.5, edgecolors='black', linewidth=0.2)

    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2],
               c='red', marker='X', s=200, label='Cluster Centers')

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("Color Clusters (RGB Space)")
    ax.legend()

    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    ax.set_zlim(0, 255)

# --- 6. New GUI Function for Color Summary ---
def _show_color_gui(all_colors_data, all_grouped_colors, cluster_centers, labels, gui_max_swatches):
    """
    Spawns a GUI to display the grouped colors and an RGB graph.
    """
    root = tk.Tk()
    root.title("Color Analysis Summary & RGB Graph")
    root.geometry("1200x800")

    paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True)

    # --- Left Pane: Grouped Colors ---
    left_frame = ttk.Frame(paned_window, padding="10")
    paned_window.add(left_frame, weight=1)

    canvas = tk.Canvas(left_frame)
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    def rgb_to_hex(rgb):
        return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'

    displayed_colors = set()

    for group_center, group_data in all_grouped_colors.items():
        colors_in_group = group_data["colors"]
        if not colors_in_group:
            continue
        
        mixed_color = group_data["mixed_color"]
        mixed_hex = rgb_to_hex(mixed_color)
        
        cluster_center_text = f"Cluster Center: {mixed_color}"
        group_label = ttk.Label(scrollable_frame, text=cluster_center_text, font=("Helvetica", 14, "bold"))
        group_label.pack(anchor="w", pady=(10, 5))
        
        mixed_frame = ttk.Frame(scrollable_frame)
        mixed_frame.pack(anchor="w", padx=10)
        
        mixed_label = ttk.Label(mixed_frame, text=f"Mixed Color: {mixed_color} ({mixed_hex})", font=("Helvetica", 10))
        mixed_label.pack(side="left")
        
        mixed_swatch = tk.Canvas(mixed_frame, width=20, height=20, bg=mixed_hex, highlightthickness=1, highlightbackground="black")
        mixed_swatch.pack(side="left", padx=5)

        swatch_frame = ttk.Frame(scrollable_frame)
        swatch_frame.pack(anchor="w", padx=10, pady=5)
        
        sorted_colors = sorted(colors_in_group, key=lambda x: x['distance'])

        displayed_count = 0
        for item in sorted_colors:
            color = item['color']
            if color in displayed_colors:
                continue
            
            if displayed_count >= gui_max_swatches:
                break
            
            count, distance = item['count'], item['distance']
            hex_color = rgb_to_hex(color)
            
            color_info = f"RGB: {color}\nHex: {hex_color}\nSim. Score: {distance:.2f}\n" \
                         f"Count: {count}"
            
            tooltip_text = color_info
            
            swatch_container = ttk.Frame(swatch_frame, borderwidth=1, relief="solid")
            swatch_container.pack(side="left", padx=5, pady=5)
            
            swatch = tk.Canvas(swatch_container, width=50, height=50, bg=hex_color, highlightthickness=1, highlightbackground="black")
            swatch.pack()

            tooltip_label = ttk.Label(swatch_container, text=tooltip_text, relief="solid",
                                      borderwidth=1, background="white", padding=5)
            tooltip_label.pack(pady=5)

            displayed_colors.add(color)
            displayed_count += 1
    
    # --- Right Pane: RGB Graph ---
    right_frame = ttk.Frame(paned_window, padding="10")
    paned_window.add(right_frame, weight=1)

    fig = plt.Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    all_colors_array = np.array([c[0] for c in all_colors_data], dtype=np.float64)
    _plot_cluster_graph(all_colors_array, cluster_centers, labels, ax)

    canvas_plot = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas_plot.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    toolbar = NavigationToolbar2Tk(canvas_plot, right_frame)
    toolbar.update()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.mainloop()

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
        nargs='?', 
        const='all',
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
        help="Number of most frequent colors to analyze from EACH image (default: 10)."
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=8,
        help="Number of color clusters to identify (default: 8)."
    )
    parser.add_argument(
        "--gui-max-swatches",
        type=int,
        default=10,
        help="Maximum number of color swatches to display per cluster in the GUI. (default: 10)"
    )

    args = parser.parse_args()

    venv_path = os.path.abspath(args.venv_name)
    requirements_file = os.path.abspath(args.reqs_file)
    db_path = "color_analysis.db"
    markdown_path = "clusters.md"

    if not any([args.setup, args.analyze_image]):
        parser.print_help()
        sys.exit(0)

    if args.setup:
        _setup_venv(venv_path, requirements_file)
        logger.info("Setup process completed.")

    if args.analyze_image:
        all_colors_data, cluster_centers, labels = _load_analysis_from_db(db_path)
        
        if all_colors_data and cluster_centers is not None and labels is not None:
            logger.info("Loaded analysis results from database.")
        else:
            logger.info("No valid analysis data found in database. Starting new analysis...")
            all_colors_data = []

            if args.analyze_image == 'all':
                logger.info(f"Analyzing all images recursively in the '{args.images_dir}' directory...")
                image_files_to_analyze = []
                for dirpath, _, filenames in os.walk(args.images_dir):
                    for filename in filenames:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                            relative_path = os.path.relpath(os.path.join(dirpath, filename), args.images_dir)
                            image_files_to_analyze.append(relative_path)
                
                if not image_files_to_analyze:
                    logger.warning("No image files found.")
                    messagebox.showinfo("No Images Found", "No image files were found in the specified directory to analyze.")
                    sys.exit(0)
                
                pool_inputs = [(path, args.images_dir, args.num_colors) for path in image_files_to_analyze]
                with multiprocessing.Pool() as pool:
                    results = pool.starmap(_analyze_image_for_clustering, pool_inputs)
                
                all_colors_data = list(itertools.chain.from_iterable(results))
            else:
                all_colors_data = _analyze_image_for_clustering(args.analyze_image, args.images_dir, args.num_colors)

            if not all_colors_data:
                messagebox.showerror("Analysis Error", "Failed to collect any color data from the images.")
                sys.exit(1)
            
            cluster_centers, labels = _cluster_all_colors(all_colors_data, args.num_clusters)
            
            _save_analysis_to_db(db_path, all_colors_data, cluster_centers, labels)

        all_grouped_colors = {tuple(center): {"mixed_color": center, "colors": []} for center in cluster_centers}
        
        for i, color_with_count in enumerate(all_colors_data):
            color, count = color_with_count
            cluster_center = tuple(cluster_centers[labels[i]])
            all_grouped_colors[cluster_center]["colors"].append({
                "color": color,
                "count": count,
                "distance": _calculate_color_distance(color, cluster_center)
            })
            
        # The fix: call the save_to_markdown function before spawning the GUI.
        _save_to_markdown(all_grouped_colors, output_path=markdown_path)
        
        _show_color_gui(all_colors_data, all_grouped_colors, cluster_centers, labels, args.gui_max_swatches)

if __name__ == "__main__":
    main()
