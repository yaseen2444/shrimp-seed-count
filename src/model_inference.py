import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def setup_cfg(config_path, weights_path, confidence_threshold=0.5):
    """
    Create config with the model parameters
    """
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.freeze()
    return cfg

def visualize_predictions(
    config_path,
    weights_path,
    test_images_dir,
    test_json_path,
    output_dir="/teamspace/studios/this_studio/shrimp-seed-counter/outputs",
    confidence_threshold=0.5
):
    """
    Visualize model predictions on test images
    
    Args:
        config_path (str): Path to model config file
        weights_path (str): Path to model weights
        test_images_dir (str): Directory containing test images
        test_json_path (str): Path to test annotations in COCO format
        output_dir (str): Directory to save visualizations
        confidence_threshold (float): Confidence threshold for predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Register the test dataset
    try:
        register_coco_instances(
            name="test_dataset",
            metadata={},
            json_file=test_json_path,
            image_root=test_images_dir
        )
    except Exception as e:
        print(f"Error registering dataset: {e}")
        return

    # Setup configuration and predictor
    cfg = setup_cfg(config_path, weights_path, confidence_threshold)
    predictor = DefaultPredictor(cfg)
    
    # Get metadata
    test_metadata = MetadataCatalog.get("test_dataset")
    
    # Get all images in the test directory
    image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg")) + \
                 glob.glob(os.path.join(test_images_dir, "*.jpeg")) + \
                 glob.glob(os.path.join(test_images_dir, "*.png"))
    
    if not image_paths:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    for image_path in image_paths:
        try:
            # Read image
            im = cv2.imread(image_path)
            if im is None:
                print(f"Could not read image: {image_path}")
                continue
                
            # Get predictions
            outputs = predictor(im)
            
            # Create visualization
            v = Visualizer(
                im[:, :, ::-1],  # BGR to RGB
                metadata=test_metadata,
                scale=0.8
            )
            
            # Draw predictions
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            # Convert for displaying/saving
            vis_image = out.get_image()[:, :, ::-1]  # RGB to BGR
            
            # Save visualization
            output_path = os.path.join(
                output_dir,
                f"pred_{os.path.basename(image_path)}"
            )
            cv2.imwrite(output_path, vis_image)
            
            # Display image
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_image[:, :, ::-1])  # BGR to RGB for display
            plt.axis('off')
            plt.title(os.path.basename(image_path))
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

def main():
    # Example usage with paths
    visualize_predictions(
        config_path='/teamspace/studios/this_studio/shrimp-seed-counter/configs/config.yml',
        weights_path='/teamspace/studios/this_studio/shrimp-seed-counter/configs/model_final.pth',
        test_images_dir='/teamspace/studios/this_studio/shrimp-seed-counter/data/test',
        test_json_path='/teamspace/studios/this_studio/shrimp-seed-counter/data/test/_annotations.coco.json',
        output_dir='/teamspace/studios/this_studio/shrimp-seed-counter/outputs',
        confidence_threshold=0.5
    )

if __name__ == "__main__":
    main()