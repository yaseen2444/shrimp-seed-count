
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

This is a link to visit out demo.(https://8503-01jff55mrhm0wpp0h15zrz0tfq.cloudspaces.litng.ai )


## Documentation

[Documentation](https://linktodocumentation)


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


## Running Tests

To run tests, run the following command

```bash
  npm run test
```


## Screenshots

![App Screenshot](https://imgur.com/undefined)


## Optimizations

- Code refactoring for improved maintainability.
- Performance improvements with optimized image processing.
- Accessibility improvements with better user interface design and navigation.
- Memory management enhancements to handle large datasets.
- UI enhancements for better user experience


