import gradio as gr
import os
import shutil
import subprocess
import threading
import queue
import time
from pathlib import Path
import trimesh

# Get the directory where this script is located (should be PartField root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model(file):
    """Load a model file and copy it to data directory in a subfolder named after the model (without extension)"""
    if file is None:
        return None, "Please upload a model file."
    
    original_filename = os.path.basename(file.name)
    model_name = Path(original_filename).stem
    
    dest_dir = os.path.join(SCRIPT_DIR, "data", model_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_path = os.path.join(dest_dir, original_filename)
    shutil.copy2(file.name, dest_path)
    
    return model_name, f"Model '{original_filename}' loaded successfully!\nSaved to: {dest_path}"


def read_output(pipe, output_queue):
    """Read output from a pipe and put it into a queue"""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                output_queue.put(line)
        pipe.close()
    except Exception as e:
        output_queue.put(f"Error reading output: {e}\n")


def run_command_streaming(cmd, cwd=None, output_buffer=""):
    """Run a command and yield output in real-time"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=cwd,
        universal_newlines=True
    )
    
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    while process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty():
        has_output = False
        
        # Check stdout
        try:
            while True:
                line = stdout_queue.get_nowait()
                output_buffer += line
                has_output = True
        except queue.Empty:
            pass
        
        # Check stderr
        try:
            while True:
                line = stderr_queue.get_nowait()
                output_buffer += line
                has_output = True
        except queue.Empty:
            pass
        
        if has_output:
            yield output_buffer
        
        if process.poll() is None:
            time.sleep(0.1)
    
    # Wait for threads and get remaining output
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    
    for queue_obj in [stdout_queue, stderr_queue]:
        try:
            while True:
                line = queue_obj.get_nowait()
                output_buffer += line
                yield output_buffer
        except queue.Empty:
            pass
    
    process.wait()
    return output_buffer


def segment_model_generator(model_name, num_clusters):
    """Run partfield_inference.py and run_part_clustering.py, yielding output in real-time"""
    if model_name is None or model_name == "":
        yield "Error: Please load a model first.", ""
        return
    
    if num_clusters < 1 or num_clusters > 100:
        yield "Error: Number of clusters must be between 1 and 100.", ""
        return
    
    output_buffer = "=" * 80 + "\n"
    output_buffer += "PARTFIELD SEGMENTATION PROCESS\n"
    output_buffer += "=" * 80 + "\n"
    yield output_buffer, ""
    
    # Paths
    model_dir = os.path.join("data", model_name)
    result_name = f"partfield_features/{model_name}"
    config_path = "configs/final/demo.yaml"
    checkpoint_path = "model/model_objaverse.ckpt"
    
    # Step 1: Run partfield_inference.py
    output_buffer += f"\n[STEP 1] Running partfield_inference.py for '{model_name}'...\n"
    output_buffer += "=" * 80 + "\n"
    yield output_buffer, ""
    
    inference_cmd = [
        "python", "-u", "partfield_inference.py",
        "-c", config_path,
        "--opts",
        "continue_ckpt", checkpoint_path,
        "result_name", result_name,
        "dataset.data_path", model_dir
    ]
    
    try:
        for buffer in run_command_streaming(inference_cmd, cwd=SCRIPT_DIR, output_buffer=output_buffer):
            yield buffer, ""
            output_buffer = buffer
        output_buffer += "\n" + "=" * 80 + "\n"
        output_buffer += "✓ Inference completed successfully!\n"
        yield output_buffer, ""
    except Exception as e:
        output_buffer += f"\n✗ Error during inference: {e}\n"
        yield output_buffer, ""
        return
    
    # Step 2: Run run_part_clustering.py
    output_buffer += "\n" + "=" * 80 + "\n"
    output_buffer += f"[STEP 2] Running run_part_clustering.py with {num_clusters} clusters...\n"
    output_buffer += "=" * 80 + "\n"
    yield output_buffer, ""
    
    root_dir = f"exp_results/partfield_features/{model_name}"
    dump_dir = f"exp_results/clustering/{model_name}"
    
    clustering_cmd = [
        "python", "-u", "run_part_clustering.py",
        "--root", root_dir,
        "--dump_dir", dump_dir,
        "--source_dir", model_dir,
        "--use_agglo", "True",
        "--max_num_clusters", str(num_clusters),
        "--option", "0"
    ]
    
    try:
        for buffer in run_command_streaming(clustering_cmd, cwd=SCRIPT_DIR, output_buffer=output_buffer):
            yield buffer, ""
            output_buffer = buffer
        output_buffer += "\n" + "=" * 80 + "\n"
        output_buffer += "✓ Clustering completed successfully!\n"
        output_buffer += f"✓ Results saved to: {dump_dir}\n"
        yield output_buffer, model_name
    except Exception as e:
        output_buffer += f"\n✗ Error during clustering: {e}\n"
        yield output_buffer, ""
        return


def convert_ply_to_glb(ply_path, glb_dir):
    """Convert a PLY file to GLB format with caching"""
    try:
        os.makedirs(glb_dir, exist_ok=True)
        
        ply_filename = os.path.basename(ply_path)
        glb_filename = os.path.splitext(ply_filename)[0] + ".glb"
        glb_path = os.path.abspath(os.path.join(glb_dir, glb_filename))
        
        # Use cache if GLB exists and is newer than PLY
        if os.path.exists(glb_path):
            if os.path.getmtime(glb_path) >= os.path.getmtime(ply_path):
                return glb_path
        
        # Convert PLY to GLB
        mesh = trimesh.load(ply_path, process=False)
        mesh.export(glb_path)
        return glb_path
    except Exception as e:
        print(f"Error converting {ply_path} to GLB: {e}")
        return None


def convert_all_ply_to_glb(model_name):
    """Convert all PLY files to GLB and return list of GLB files"""
    if model_name is None or model_name == "":
        return []
    
    ply_dir = os.path.join(SCRIPT_DIR, "exp_results", "clustering", model_name, "ply")
    glb_dir = os.path.join(SCRIPT_DIR, "exp_results", "clustering", model_name, "glb")
    
    if not os.path.exists(ply_dir):
        return []
    
    glb_files = []
    for filename in sorted(os.listdir(ply_dir)):
        if filename.endswith(".ply"):
            ply_path = os.path.abspath(os.path.join(ply_dir, filename))
            if os.path.exists(ply_path):
                glb_path = convert_ply_to_glb(ply_path, glb_dir)
                if glb_path:
                    glb_filename = os.path.basename(glb_path)
                    glb_files.append((glb_filename, glb_path))
    
    return glb_files


def get_all_glb_files():
    """Get all GLB files from all previous segmentations"""
    clustering_dir = os.path.join(SCRIPT_DIR, "exp_results", "clustering")
    
    if not os.path.exists(clustering_dir):
        return []
    
    all_glb_files = []
    
    # Iterate through all model directories
    for model_name in sorted(os.listdir(clustering_dir)):
        model_dir = os.path.join(clustering_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        
        glb_dir = os.path.join(model_dir, "glb")
        if not os.path.exists(glb_dir):
            continue
        
        # Check if GLB files exist, if not, try to convert from PLY
        glb_files_in_dir = [f for f in os.listdir(glb_dir) if f.endswith(".glb")]
        
        if not glb_files_in_dir:
            # Try to convert PLY files to GLB
            convert_all_ply_to_glb(model_name)
            glb_files_in_dir = [f for f in os.listdir(glb_dir) if f.endswith(".glb")]
        
        # Add all GLB files from this model
        for glb_filename in sorted(glb_files_in_dir):
            glb_path = os.path.abspath(os.path.join(glb_dir, glb_filename))
            if os.path.exists(glb_path):
                all_glb_files.append({
                    "name": f"{model_name}/{glb_filename}",
                    "path": glb_path
                })
    
    return all_glb_files


# Create Gradio interface
with gr.Blocks(title="PartField Segmentation") as app:
    gr.Markdown("# PartField Model Segmentation")
    gr.Markdown("Upload a 3D model and segment it into parts using PartField.")
    
    model_name_state = gr.State(value=None)
    
    with gr.Tabs() as tabs:
        # Tab 1: Generation
        with gr.Tab("Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="Upload Model File",
                        file_types=[".glb", ".obj", ".ply"]
                    )
                    num_clusters = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Number of Clusters"
                    )
                    segment_btn = gr.Button("Segment", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=1):
                    terminal_output = gr.Textbox(
                        label="Terminal Output",
                        lines=25,
                        interactive=False,
                        show_copy_button=True,
                        autoscroll=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Model Viewer")
                    model_dropdown = gr.Dropdown(
                        label="Select Segmentation Result",
                        choices=[],
                        value=None,
                        interactive=True
                    )
                    model_viewer = gr.Model3D(
                        label="3D Model",
                        show_label=True
                    )
        
        # Tab 2: Gallery of all models
        with gr.Tab("Model Gallery"):
            refresh_btn = gr.Button("Refresh Gallery", variant="secondary")
            gallery_info = gr.Markdown("Click 'Refresh Gallery' to load all models.")
            
            # Create grid of Model3D viewers (up to 50 models)
            gallery_viewers = []
            max_viewers = 50
            
            for i in range(0, max_viewers, 2):
                with gr.Row():
                    for j in range(2):
                        idx = i + j
                        if idx < max_viewers:
                            with gr.Column():
                                viewer_label = gr.Markdown("", visible=False)
                                viewer = gr.Model3D(visible=False, show_label=False)
                                gallery_viewers.append((viewer_label, viewer))
    
    def populate_gallery(max_viewers):
        """Populate gallery with all models"""
        all_glb_files = get_all_glb_files()
        
        if not all_glb_files:
            # Hide all viewers and show message
            updates = [gr.Markdown("No models found. Generate some segmentations first!", visible=True)]
            for i in range(max_viewers):
                updates.append(gr.Markdown(visible=False))
                updates.append(gr.Model3D(visible=False))
            return updates
        
        num_models = len(all_glb_files)
        updates = []
        
        # Update info
        updates.append(gr.Markdown(f"### Displaying {num_models} models", visible=True))
        
        # Update each viewer
        for i in range(max_viewers):
            if i < num_models:
                item = all_glb_files[i]
                updates.append(gr.Markdown(f"**{item['name']}**", visible=True))
                updates.append(gr.Model3D(value=item["path"], visible=True))
            else:
                updates.append(gr.Markdown(visible=False))
                updates.append(gr.Model3D(visible=False))
        
        return updates
    
    def segment_and_update(file, num_clusters):
        """Handle segmentation button click - runs segmentation and prepares GLB files"""
        if file is None:
            yield "Error: Please upload a model file first.", gr.Dropdown(choices=[], value=None), None, None
            return
        
        # Load the model
        model_name, load_message = load_model(file)
        if model_name is None:
            yield load_message, gr.Dropdown(choices=[], value=None), None, None
            return
        
        # Run segmentation with streaming output
        completed_model_name = ""
        terminal_output_text = ""
        
        for terminal_output_text, completed_model_name in segment_model_generator(model_name, num_clusters):
            yield terminal_output_text, gr.Dropdown(choices=[], value=None), None, completed_model_name or None
        
        # After completion, convert all PLY files to GLB
        if completed_model_name:
            time.sleep(0.5)  # Small delay to ensure files are fully written
            glb_files = convert_all_ply_to_glb(completed_model_name)
            
            if glb_files:
                choices = [f[0] for f in glb_files]
                first_glb_path = glb_files[0][1]
                yield terminal_output_text, gr.Dropdown(choices=choices, value=choices[0]), first_glb_path, completed_model_name
            else:
                yield terminal_output_text, gr.Dropdown(choices=[], value=None), None, completed_model_name
        else:
            yield terminal_output_text, gr.Dropdown(choices=[], value=None), None, None
    
    def update_viewer_from_dropdown(selected_glb, stored_model_name):
        """Update the model viewer when a different GLB file is selected"""
        if selected_glb is None or stored_model_name is None or selected_glb == "":
            return None
        
        glb_path = os.path.join(SCRIPT_DIR, "exp_results", "clustering", stored_model_name, "glb", selected_glb)
        glb_path = os.path.abspath(glb_path)
        
        if os.path.exists(glb_path):
            return glb_path
        return None
    
    
    segment_btn.click(
        fn=segment_and_update,
        inputs=[file_input, num_clusters],
        outputs=[terminal_output, model_dropdown, model_viewer, model_name_state]
    )
    
    model_dropdown.change(
        fn=update_viewer_from_dropdown,
        inputs=[model_dropdown, model_name_state],
        outputs=model_viewer
    )
    
    # Flatten gallery_viewers for outputs
    gallery_outputs = []
    for label, viewer in gallery_viewers:
        gallery_outputs.extend([label, viewer])
    
    refresh_btn.click(
        fn=lambda: populate_gallery(max_viewers),
        inputs=[],
        outputs=[gallery_info] + gallery_outputs
    )
    
    # Load gallery on app start
    app.load(
        fn=lambda: populate_gallery(max_viewers),
        inputs=[],
        outputs=[gallery_info] + gallery_outputs
    )

if __name__ == "__main__":
    app.launch()
