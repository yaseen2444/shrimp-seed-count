## Shrimp-Seed-Counter 

- This project focuses on detecting prawns-seeds in images using a Faster R-CNN model with a ResNet backbone. Designed for aquaculture applications, it leverages cutting-edge deep learning techniques like region proposal networks and Slicing-Aided Hyper Inference (SAHI) to achieve accurate detection, even for small objects. The project is ideal for researchers and developers in computer vision and aquaculture, seeking to automate prawn-seed detection with high precision.


## Acknowledgements

- This project was made possible by the efforts of the Computer Vision Team. Special thanks to:

- Roboflow for providing the prawn dataset for dataset management.(https://roboflow.com/)

- Developers and contributors of SAHI (Slicing Aided  Hyper Inference) for improving small object detection capabilities. (https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/)
- COCO Dataset contributors for pre-trained weights that enabled effective transfer learning.
(https://universe.roboflow.com/yaseen-vgnae/ shrimp-counting-j5era)

- PyTorch for the deep learning framework used in model implementation.(https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

- How to Train Detectron2 on Custom Object(Blog).
(https://colab.research.google.com/drive/1-TNOcPm3Jr3fOJG8rnGT9gh60mHUsvaW)

- Faster R-CNN Paper.(https://arxiv.org/abs/1506.01497)
- ResNet Paper.(https://arxiv.org/abs/1512.03385)


## Authors

- [@Yaseenmohammad](https://github.com/yaseen2444)

- [@rishendra-manne](https://github.com/rishendra-manne)

## Demo

This is a link to visit out demo.(https://www.youtube.com/watch?v=StxUNM17PMw)


# Comprehensive Code Explanation

## 1. Trainer Code (`trainer.py`)

### DatasetMapper Class
```python
class DatasetMapper:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def __call__(self, dataset_dict):
        # Deep copy to avoid modifying original data
        dataset_dict = copy.deepcopy(dataset_dict)
```
**Purpose**: Custom data preprocessing for object detection
- Creates a deep copy of dataset to prevent unintended modifications
- Prepares images and annotations for model training

#### Image Preprocessing
```python
image = utils.read_image(dataset_dict["file_name"], format="BGR")
preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```
**Key Steps**:
- Reads image in BGR format (OpenCV default)
- Converts image to RGB color space
- Ensures compatibility with model input requirements

#### Image Transformations
```python
transform_list = [
    T.Resize((800, 600)),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
]
```
**Augmentation Techniques**:
- Resizes images to consistent 800x600 dimension
- Applies random horizontal flipping (50% probability)
- Increases model's robustness and generalization

### ShrimpTrainer Class
```python
class ShrimpTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)
```
**Purpose**: 
- Customizes training data loading process
- Applies custom DatasetMapper during training
- Enables advanced data preprocessing

### ModelConfig Class
```python
class ModelConfig:
    def __init__(self, num_classes=2, base_lr=0.002, max_iter=2500):
        self.cfg = get_cfg()
        self.num_classes = num_classes
        self.base_lr = base_lr
        self.max_iter = max_iter
```
**Configuration Parameters**:
- `num_classes`: Number of object classes (default: 2)
- `base_lr`: Initial learning rate
- `max_iter`: Maximum training iterations

#### Model Setup Method
```python
def setup(self):
    # Load Faster R-CNN configuration
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
```
**Key Configuration Steps**:
- Uses pre-trained Faster R-CNN with ResNet-50 backbone
- Loads COCO-trained weights for transfer learning

### Training Pipeline
```python
class TrainingPipeline:
    def register_datasets(self):
        register_coco_instances(
            "my_dataset_train", 
            {}, 
            str(self.train_json), 
            str(self.data_dir / "train")
        )
```
**Dataset Registration**:
- Registers training and validation datasets
- Uses COCO annotation format
- Enables Detectron2's dataset management

## 2. Inference Code (`inference.py`)

### Configuration Setup
```python
def setup_cfg(config_path, weights_path, confidence_threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
```
**Purpose**:
- Loads model configuration
- Sets model weights
- Defines confidence threshold for predictions

### Prediction Visualization
```python
def visualize_predictions(config_path, weights_path, test_images_dir, test_json_path):
    # Register test dataset
    register_coco_instances(
        name="test_dataset",
        metadata={},
        json_file=test_json_path,
        image_root=test_images_dir
    )
```
**Key Steps**:
- Registers test dataset
- Prepares for model inference
- Supports multiple image formats

## 3. Streamlit Frontend (`app.py`)

### GPU Detection
```python
def check_gpu():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        st.sidebar.success(f"Using GPU: {device}")
        return True
```
**Features**:
- Checks GPU availability
- Displays GPU information
- Enables GPU/CPU switching

### Image Processing Function
```python
def process_image(image, model, slice_size, overlap_ratio, iou_threshold, max_area):
    def slicer_callback(slice: np.ndarray) -> sv.Detections:
        # Performs detection on image slices
        outputs = model(slice)
        instances = outputs["instances"].to("cpu")
```
**Advanced Detection Techniques**:
- Implements image slicing
- Processes image in smaller sections
- Handles complex image scenarios

### ROI Mask Creation
```python
def create_roi_mask(image_shape, polygon):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask
```
**Purpose**:
- Creates region of interest (ROI) mask
- Focuses detection on specific image regions
- Improves precision by limiting detection area

## 🚀 About Me


---

I’m Yaseen Mohammad, a passionate tech enthusiast specializing in machine learning, deep learning, and AI-driven solutions. With a strong foundation in Python programming and ML, I have hands-on experience building impactful projects across domains like computer vision, traffic management, and safety technology.  

### Highlights  
- **Key Projects**:  
- Prawn seed detection using Faster R-CNN.  
- Smart Traffic Management using YOLO and RL for dynamic signal control.  
- Speech-enabled safety app for women and child safety leveraging ASR and AI-generated context analysis.  
- SUMO-based traffic simulations integrated with PPO and DeePPO algorithms for optimization.  

- **Innovative Concepts**:  
- Retrieval-Augmented Generation (RAG) chatbots with RLHF integration.  
- Short-term memory buffer ASR systems for emergency response applications.  
- Inference slicing for small object detection in high-density images.  

- **Problem-Solving**:  
- Addressed real-world challenges during hackathons, including Smart India Hackathon and college-level competitions.  
- Designed creative, scalable solutions blending AI with practical needs, such as traffic systems and safety apps.  

### Goals  
I am committed to leveraging AI to solve real-world problems, making technology accessible and impactful. My aim is to create innovative tools that enhance workflows, improve safety, and address pressing global challenges.  

--- 



## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/yaseen-mohammad-0a77992b1)


## Features
- GPU Utilization Check:

Checks if a GPU is available.
Displays GPU details such as device name, CUDA version, and memory if available.
- Model Loading:

Loads a Detectron2 model with specified configurations and weights.
Automatically selects GPU or CPU for inference based on availability.
- Region of Interest (ROI):

Allows users to define a polygonal ROI in the image.
Displays an interactive preview of the ROI with adjustable color and opacity.
- Slicing and Detection:

Slices the image into smaller sections for processing.
Processes each slice with the object detection model.
Merges overlapping bounding boxes to refine detections.
- Bounding Box Filtering:

Removes bounding boxes that exceed a specified area to exclude unwanted detections.
Supports user-adjustable IoU thresholds to control box merging.
- User Input Options:

Users can upload an image or select from preloaded sample images.
Provides a text area to input custom ROI coordinates and templates for quick ROI setup.
- Adjustable Parameters:

Allows users to adjust slicing size, overlap ratio, IoU threshold, and maximum bounding box area for fine-tuning detection performance.
- Real-Time Feedback:

Displays validation results for ROI coordinates.
Provides success/error messages to guide users.

## 🛠 Skills
- Python.
- Machine-learning.
- Python data analysis -tools.
- Python front-end tools.. 


## License

- This project is proprietary and may not be used, modified, or distributed without explicit permission from the project owner.

## Related

Here are some related projects

[Project Repo](https://github.com/yaseen2444/book-rec.git)

## Screenshots of working of project.

![Input - Outpu](Screenshot_26-1-2025_102939_.jpeg?width=600)

## Optimizations

- Code refactoring for improved maintainability.
- Performance improvements with optimized image processing.
- Accessibility improvements with better user interface design and navigation.
- Memory management enhancements to handle large datasets.
- UI enhancements for better user experience


