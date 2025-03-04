import pandas as pd
import os
from pathlib import Path
import base64
from PIL import Image
import io

# Configuration
csv_path = '/home/lukasculac/Desktop/Projekti/LUMEN/new_training_duplicates.csv'  # Path to your CSV file
image_dir = '/home/lukasculac/Desktop/Projekti/LUMEN/train_224x224/'  # Path to your image directory
output_html = 'image_pairs_viewer.html'  # Output HTML file
limit = 100  # Number of image pairs to display

# Create HTML templates
html_header = """
<!DOCTYPE html>
<html>
<head>
    <title>Similar Image Pairs Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .pair-container {
            display: flex;
            margin: 20px auto;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            max-width: 900px;
        }
        .image-container {
            flex: 1;
            padding: 10px;
            text-align: center;
            border: 1px solid #eee;
            margin: 0 10px;
            background-color: #fafafa;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }
        .image-id {
            margin-top: 10px;
            font-size: 14px;
            color: #555;
            word-break: break-all;
        }
        .pair-number {
            text-align: center;
            font-weight: bold;
            margin-bottom: 5px;
            color: #333;
        }
        .metrics {
            margin-top: 10px;
            font-size: 12px;
            color: #777;
            text-align: left;
        }
    </style>
</head>
<body>
    <h1>Similar Image Pairs Viewer</h1>
"""

html_footer = """
</body>
</html>
"""

pair_template = """
    <div class="pair-container">
        <div class="pair-number">Pair {pair_num}</div>
        <div class="image-container">
            <img src="data:image/jpeg;base64,{img1_base64}" alt="{img1_id}">
            <div class="image-id">{img1_id}</div>
        </div>
        <div class="image-container">
            <img src="data:image/jpeg;base64,{img2_base64}" alt="{img2_id}">
            <div class="image-id">{img2_id}</div>
        </div>
    </div>
"""

def image_to_base64(image_path):
    """Convert an image to base64 for embedding in HTML"""
    try:
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return a placeholder for missing images
        return ""

def main():
    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} image pairs from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
    
    # Limit to the first N pairs
    df = df.head(limit)
    
    # Initialize HTML content
    html_content = html_header
    
    # Process each pair
    pairs_added = 0
    
    for idx, row in df.iterrows():
        try:
            # Get image IDs
            img1_id = row['image_name_1']
            img2_id = row['image_name_2']
            
            # Construct image paths
            img1_path = os.path.join(image_dir, f"{img1_id}.jpg")
            img2_path = os.path.join(image_dir, f"{img2_id}.jpg")
            
            # Check if images exist
            if not os.path.exists(img1_path):
                print(f"Warning: Image not found: {img1_path}")
                continue
                
            if not os.path.exists(img2_path):
                print(f"Warning: Image not found: {img2_path}")
                continue
                
            # Convert images to base64
            img1_base64 = image_to_base64(img1_path)
            img2_base64 = image_to_base64(img2_path)
            
            if not img1_base64 or not img2_base64:
                print(f"Skipping pair {idx+1} due to image conversion error")
                continue
                
            # Add pair to HTML
            html_content += pair_template.format(
                pair_num=idx+1,
                img1_id=img1_id,
                img2_id=img2_id,
                img1_base64=img1_base64,
                img2_base64=img2_base64
            )
            
            pairs_added += 1
            print(f"Added pair {pairs_added}: {img1_id} and {img2_id}")
            
        except Exception as e:
            print(f"Error processing pair {idx+1}: {e}")
            continue
    
    # Add footer
    html_content += html_footer
    
    # Write HTML to file
    with open(output_html, 'w') as f:
        f.write(html_content)
    
    print(f"Done! Created HTML viewer with {pairs_added} image pairs.")
    print(f"Open {output_html} in your browser to view the images.")

if __name__ == "__main__":
    main()