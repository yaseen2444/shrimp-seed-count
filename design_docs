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
