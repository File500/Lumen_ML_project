# Updated Isaac Sim Synthetic Image Generator
# Adapted for modern Isaac Sim API (4.5+)

import os
import sys
import random
import time
import datetime
import traceback

# Create a log file that we know we can write to
log_file = os.path.join(os.path.expanduser("~"), "Desktop", "isaac_synthetic_images.log")


# Log function that writes to both console and file
def log_message(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(formatted_message + "\n")


# Clear indicator that our script is running
log_message("=" * 80)
log_message("SYNTHETIC IMAGE GENERATOR STARTING")
log_message("=" * 80)

try:
    # Import SimulationApp first (required before other Isaac imports)
    log_message("Importing simulation components...")
    from omni.isaac.kit import SimulationApp

    # Initialize the simulation application
    log_message("Initializing simulation...")
    simulation_app = SimulationApp({
        "headless": True,  # Set to True for no GUI
        "width": 1280,
        "height": 720,
    })

    # Only import other modules after SimulationApp is initialized
    log_message("Importing additional modules...")
    import numpy as np
    import omni.replicator.core as rep
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.stage import add_reference_to_stage


    def generate_synthetic_images(input_folder, output_folder, variations_per_image=10):
        """
        Generate synthetic variations of all images in a folder using NVIDIA Isaac Sim.

        Args:
            input_folder (str): Path to folder containing input images
            output_folder (str): Path to save synthetic images
            variations_per_image (int): Number of synthetic variations to generate per input image
        """
        # Start timing the entire process
        total_start_time = time.time()

        log_message(f"Input folder: {input_folder}")
        log_message(f"Output folder: {output_folder}")
        log_message(f"Variations per image: {variations_per_image}")

        # Ensure output directory exists
        try:
            os.makedirs(output_folder, exist_ok=True)
            log_message(f"Output directory confirmed: {output_folder}")
        except Exception as e:
            log_message(f"ERROR: Failed to create output directory: {str(e)}")
            return

        # Get list of image files in the input folder
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = [f for f in os.listdir(input_folder)
                           if os.path.isfile(os.path.join(input_folder, f)) and
                           os.path.splitext(f.lower())[1] in image_extensions]

            if not image_files:
                log_message(f"WARNING: No image files found in {input_folder}")
                return

            log_message(f"Found {len(image_files)} images in {input_folder}")
        except Exception as e:
            log_message(f"ERROR: Failed to read input directory: {str(e)}")
            return

        # Track overall progress
        total_images = len(image_files)
        total_variations = total_images * variations_per_image
        variations_completed = 0

        # Process each image
        for img_index, img_file in enumerate(image_files):
            img_start_time = time.time()
            img_path = os.path.join(input_folder, img_file)
            img_name = os.path.splitext(img_file)[0]

            log_message(f"Processing image {img_index + 1}/{total_images}: {img_file}")

            try:
                # Create a new scene for each image
                rep.create.scene()
                log_message(f"  - Created new scene")

                # Create a simple plane to project the image onto
                plane = rep.create.plane(
                    position=(0, 0, 0),
                    scale=(3, 3, 1),
                    semantics=[("class", "reference_image")]
                )
                log_message(f"  - Created plane")

                # Apply the input image as a texture to the plane
                texture = rep.create.texture_2d(path=img_path)
                rep.modify.material(plane, diffuse_texture=texture)
                log_message(f"  - Applied image texture")

                # Create camera
                camera = rep.create.camera(
                    position=(0, -5, 1.5),
                    look_at=(0, 0, 0),
                    focal_length=24.0
                )
                log_message(f"  - Created camera")

                # Set up randomizers for generating variations
                light_rig = rep.create.light_rig(
                    position=(0, 0, 5),
                    intensity=750,
                    temperature=6500
                )
                log_message(f"  - Created lighting")

                # Create output directory for this image
                img_output_dir = os.path.join(output_folder, img_name)
                os.makedirs(img_output_dir, exist_ok=True)

                # Create a writer to save the output images
                writer = rep.WriterRegistry.get("OmniWriter")
                writer.initialize(
                    output_dir=img_output_dir,
                    rgb=True,
                    semantic_segmentation=True,
                    depth=True,
                    normals=True
                )
                log_message(f"  - Set up writer to {img_output_dir}")

                # Generate multiple variations for this image
                log_message(f"  - Starting generation of {variations_per_image} variations")

                # Hook into the frame update to track progress
                frame_counter = [0]  # Use list for mutable reference

                def frame_update_callback():
                    frame_counter[0] += 1
                    log_message(f"    - Generated variation {frame_counter[0]}/{variations_per_image}")

                # Register the callback
                rep.event.on_frame_update(frame_update_callback)

                with rep.trigger.on_frame(num_frames=variations_per_image):
                    # Randomize lighting
                    rep.randomizer.light_intensity(light_rig, intensity_range=(500, 1000))
                    rep.randomizer.light_temperature(light_rig, temperature_range=(5000, 7500))

                    # Randomize camera position slightly for different perspectives
                    rep.randomizer.camera_position(
                        camera,
                        position_range=[(-1, -5.5, 1), (1, -4.5, 2)]
                    )

                    # Randomize camera rotation slightly
                    rep.randomizer.camera_rotation(
                        camera,
                        rotation_range=[(-5, -5, -5), (5, 5, 5)]
                    )

                    # Add some post-processing effects
                    rep.randomizer.post_processing(
                        vignette_intensity_range=(0.0, 0.3),
                        grain_intensity_range=(0.0, 0.2),
                        chromatic_aberration_range=(0.0, 0.2)
                    )

                    # Optional: add random environmental objects around the image
                    if random.random() > 0.5:
                        # Add random objects from a library if available
                        assets_root = get_assets_root_path()
                        if assets_root:
                            random_props = ["Props/Shape/Cube.usd", "Props/Shape/Sphere.usd"]
                            prop_path = os.path.join(assets_root, random.choice(random_props))
                            prop_ref = add_reference_to_stage(prop_path)

                            # Position the prop randomly in the scene
                            x = random.uniform(-3, 3)
                            y = random.uniform(-2, 2)
                            z = random.uniform(0, 2)
                            rep.modify.pose(prop_ref, position=(x, y, z))

                # Run simulation for enough steps to generate all variations
                log_message(f"  - Running simulation steps...")
                for _ in range(variations_per_image + 5):  # Add a few extra steps for safety
                    simulation_app.update()
                    time.sleep(0.1)  # Small delay to ensure processing completes

                # Unregister the callback
                rep.event.remove_on_frame_update(frame_update_callback)

                variations_completed += variations_per_image
                img_end_time = time.time()
                img_duration = img_end_time - img_start_time

                log_message(f"✓ Completed image {img_index + 1}/{total_images}: {img_file}")
                log_message(f"  - Processing time: {img_duration:.2f} seconds")
                log_message(
                    f"  - Overall progress: {variations_completed}/{total_variations} variations ({(variations_completed / total_variations) * 100:.1f}%)")

                # Estimate remaining time
                if img_index > 0:  # Only estimate after at least one image
                    avg_time_per_image = img_duration
                    images_remaining = total_images - (img_index + 1)
                    est_time_remaining = avg_time_per_image * images_remaining
                    est_completion_time = datetime.datetime.now() + datetime.timedelta(seconds=est_time_remaining)
                    log_message(f"  - Estimated time remaining: {est_time_remaining:.1f} seconds")
                    log_message(f"  - Estimated completion: {est_completion_time.strftime('%H:%M:%S')}")

            except Exception as e:
                log_message(f"ERROR processing {img_file}: {str(e)}")
                log_message(traceback.format_exc())
                # Continue with the next image
                continue

        # Cleanup resources
        try:
            writer.close()
            log_message("Closed writer resources")
        except Exception as e:
            log_message(f"WARNING: Error during cleanup: {str(e)}")

        # Calculate total time
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        log_message(f"=== COMPLETED SYNTHETIC IMAGE GENERATION ===")
        log_message(f"Total processing time: {total_duration:.2f} seconds ({total_duration / 60:.2f} minutes)")
        log_message(f"Successfully generated {variations_completed} variations from {total_images} input images")
        log_message(f"Output saved to: {output_folder}")


    # Run the generator with your paths
    # Make sure these are absolute paths to avoid confusion
    input_folder = "D:/NotebookPy/Lumen_ML_project/Lumen_Image_Data/malignant"
    output_folder = "D:/NotebookPy/Lumen_ML_project/Lumen_Image_Data/synthetic_images"

    log_message("Starting synthetic image generation...")
    generate_synthetic_images(
        input_folder=input_folder,
        output_folder=output_folder,
        variations_per_image=10
    )
    log_message("Image generation completed!")

except Exception as e:
    log_message(f"CRITICAL ERROR: {str(e)}")
    log_message(traceback.format_exc())

finally:
    # Make sure we cleanup properly
    log_message("Shutting down Isaac Sim...")
    try:
        simulation_app.close()
        log_message("Isaac Sim closed successfully")
    except:
        log_message("Error closing Isaac Sim (may not have been initialized)")

    log_message("SCRIPT EXECUTION COMPLETE")