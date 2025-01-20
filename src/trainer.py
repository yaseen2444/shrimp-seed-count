import os
import copy
import torch
import cv2
import numpy as np
from pathlib import Path
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances

class DatasetMapper:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Read and preprocess image
        try:
            image = utils.read_image(dataset_dict["file_name"], format="BGR")
        except Exception as e:
            print(f"Error reading image {dataset_dict['file_name']}: {e}")
            raise
            
        # Convert to RGB
        preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        transform_list = [
            T.Resize((800, 600)),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        ]
        
        try:
            preprocessed_image, transforms = T.apply_transform_gens(transform_list, preprocessed_image)
            dataset_dict["image"] = torch.as_tensor(preprocessed_image.transpose(2, 0, 1).astype("float32"))
        except Exception as e:
            print(f"Error in image transformation: {e}")
            raise
            
        # Handle annotations
        try:
            annos = [
                utils.transform_instance_annotations(obj, transforms, preprocessed_image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, preprocessed_image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        except Exception as e:
            print(f"Error processing annotations: {e}")
            raise
            
        return dataset_dict

class ShrimpTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg)
        return build_detection_train_loader(cfg, mapper=mapper)

class ModelConfig:
    def __init__(self, num_classes=2, base_lr=0.002, max_iter=2500):
        self.cfg = get_cfg()
        self.num_classes = num_classes
        self.base_lr = base_lr
        self.max_iter = max_iter
        
    def setup(self):
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        
        # Dataset settings
        self.cfg.DATASETS.TRAIN = ("my_dataset_train",)
        self.cfg.DATASETS.TEST = ("my_dataset_valid",)
        
        # Training settings
        self.cfg.DATALOADER.NUM_WORKERS = 4
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.cfg.SOLVER.IMS_PER_BATCH = 8
        self.cfg.SOLVER.BASE_LR = self.base_lr
        self.cfg.SOLVER.MAX_ITER = self.max_iter
        self.cfg.SOLVER.STEPS = [int(self.max_iter * 0.6), int(self.max_iter * 0.8)]
        self.cfg.SOLVER.CHECKPOINT_PERIOD = int(self.max_iter / 5)
        
        # Model settings
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        self.cfg.MODEL.BACKBONE.FREEZE_AT = 2
        
        return self.cfg

class TrainingPipeline:
    def __init__(self, data_dir, output_dir="output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_json = self.data_dir / "train/_annotations.coco.json"
        self.valid_json = self.data_dir / "valid/_annotations.coco.json"
        
    def register_datasets(self):
        """Register the training and validation datasets"""
        if not self.train_json.exists() or not self.valid_json.exists():
            raise FileNotFoundError("Dataset annotation files not found")
            
        register_coco_instances(
            "my_dataset_train", 
            {}, 
            str(self.train_json), 
            str(self.data_dir / "train")
        )
        register_coco_instances(
            "my_dataset_valid", 
            {}, 
            str(self.valid_json), 
            str(self.data_dir / "valid")
        )
        
    def train(self, num_classes=2, base_lr=0.002, max_iter=2500):
        """Run the training pipeline"""
        try:
            # Register datasets
            self.register_datasets()
            
            # Setup configuration
            model_config = ModelConfig(num_classes, base_lr, max_iter)
            cfg = model_config.setup()
            cfg.OUTPUT_DIR = str(self.output_dir)
            os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
            
            # Initialize trainer
            trainer = ShrimpTrainer(cfg)
            trainer.resume_or_load(resume=False)
            
            # Start training
            print("Starting training...")
            trainer.train()
            
            # Save configuration
            config_path = self.output_dir / "config.yml"
            with open(config_path, "w") as f:
                f.write(cfg.dump())
            print(f"Configuration saved to {config_path}")
            
            return cfg.OUTPUT_DIR
            
        except Exception as e:
            print(f"Training failed: {e}")
            raise

def main():
    # Set your data directory path
    data_dir = "/teamspace/studios/this_studio/shrimp-seed-counter/data"
    output_dir = "/teamspace/studios/this_studio/shrimp-seed-counter/configs"
    
    try:
        pipeline = TrainingPipeline(data_dir, output_dir)
        output_path = pipeline.train()
        print(f"Training completed successfully. Model saved in {output_path}")
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    main()