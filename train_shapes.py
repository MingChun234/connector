import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import subprocess

import tensorflow as tf
from keras.callbacks import EarlyStopping, Callback
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from pycocotools.coco import COCO

# 設置 TensorFlow 日誌等級
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 設置 GPU 記憶體動態增長
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 設置 GPU 動態增長記憶體
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Set memory growth for GPU successfully.")
    except RuntimeError as e:
        print(f"Runtime error while setting memory growth: {e}")

# 列出所有可用的物理設備
devices = tf.config.experimental.list_physical_devices()
print("Local devices available:")
for device in devices:
    print(device)

# 確認是否有 GPU
gpu = tf.config.experimental.list_physical_devices('GPU')
if gpu:
    print("GPU is available.")
else:
    print("No GPU available.")

# 使用當前工作目錄作為基礎
MODEL_DIR = os.path.join(os.getcwd(), "logs")
# print("MODEL_DIR:", MODEL_DIR)

# 確保路徑存在
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Root directory of the project
ROOT_DIR = os.getcwd()


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ConnectorConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "connector"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 2 shapes(connector、car)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image beFcause the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 40

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 13

config = ConnectorConfig()
config.display()

"""## Notebook Preferences"""

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

"""## Dataset

Create a synthetic dataset

Extend the Dataset class and add a method to load the connector dataset, `load_connector()`, and override the following methods:

* load_image()
* load_mask()
* image_reference()
"""

class ConnectorDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_connector(self, dataset_dir):
        """Load the connector dataset from COCO format."""
        print("正在讀取 COCO 註釋...")
        coco = COCO(os.path.join(dataset_dir, "_annotations.coco.json"))
        print("註釋讀取完成。")

        class_ids = sorted(coco.getCatIds())
        print("類別 ID:", class_ids)

        image_ids = list(coco.imgs.keys())
        print("讀取到的圖片 ID:", image_ids)
        
        for image_id in image_ids:
            image_info = coco.loadImgs(image_id)[0]
            print("讀取圖片資訊:", image_info)

            annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None))
            # print(f"Image ID {image_id} annotations: {annotations}")

            self.add_image(
                "connector",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_info['file_name']),
                width=image_info["width"],
                height=image_info["height"],
                annotations=annotations
            )
            print("已添加圖片 ID:", image_id)

    def load_image(self, image_id):
        #從圖像ID生成圖像，從文件加載實際圖像
        info = self.image_info[image_id]
        image_path = os.path.join(info['path'])
        print(image_path)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        avg_color_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_row, axis=0)

        bg_color = avg_color.astype(np.uint8)
        bg_image = np.ones([info['height'], info['width'], 3], dtype=np.uint8) * bg_color
        
        return bg_image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "connector":
            return info["connector"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        
        for i, annotation in enumerate(annotations):
            # 提取 segmentation 或 bbox 資料來生成 mask
            segmentation = annotation['segmentation']
            mask[:, :, i] = self.draw_segmentation(mask[:, :, i].copy(), segmentation)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([annotation['category_id'] for annotation in annotations])
        return mask.astype(bool), class_ids.astype(np.int32)

    def draw_segmentation(self, mask, segmentation):
        """Draw segmentation mask from segmentation data."""
        for segment in segmentation:
            poly = np.array(segment).reshape((-1, 2))  # Reshape segmentation points
            cv2.fillPoly(mask, [np.int32(poly)], 1)    # Fill the polygon on the mask
        return mask
    

# Training dataset
dataset_train = ConnectorDataset()
dataset_train.add_class("connector", 1, "connector")
dataset_train.add_class("connector", 2, "car")
dataset_train.load_connector("connector_dataset/train")
dataset_train.prepare()

# Validation dataset
dataset_val = ConnectorDataset()
dataset_val.add_class("connector", 1, "connector")
dataset_val.add_class("connector", 2, "car")
dataset_val.load_connector("connector_dataset/valid")
dataset_val.prepare()


# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    print("class_ids:", class_ids)
    print("class_names:", dataset_train.class_names)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


"""## Create Model"""

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

"""## Training

Train in two stages:
1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.

2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.
""" 

# train_generator = modellib.data_generator(dataset_train, config, shuffle=True)
# data = next(train_generator)
# print(data)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.


# 建立可視化圖表類別 用義:記錄每次epoch的歷史紀錄

class visualizegraph(Callback):
    def __init__(self, save_dir, val_data):
        super(visualizegraph, self).__init__()
        self.save_dir = save_dir
        self.val_data = val_data
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # 畫損失圖表
    def loss_graph(self, epoch, epoch_dir):

        # 使用 Keras 自動保存的歷史數據
        val_loss_history = self.model.history.history.get('val_loss', [])  
        train_loss_history = self.model.history.history.get('loss', [])      
        
        fig, ax = plt.subplots()
        
        # 繪製從第 1 回合到當前回合的歷史損失
        ax.plot(range(1, epoch + 2), val_loss_history, label='Val Loss', color='blue')
        ax.plot(range(1, epoch + 2), train_loss_history, label='Train Loss', color='red')
        
        ax.set_title(f'Epoch {epoch + 1} - Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()

        plt.savefig(os.path.join(epoch_dir, f'epoch-{epoch + 1:02d}-loss.png'))
        plt.close(fig)

    # 畫物件檢測圖表(包括邊界框、遮罩和類別和置信度分數)
    def objdet_graph(self, epoch, epoch_dir):

        image_id = random.choice(self.val_data.image_ids)
        original_image = self.val_data.load_image(image_id)

        r = self.model.detect([original_image], verbose=0)[0]

        fig, ax = plt.subplots()

        # 顯示檢測結果，包括邊界框、遮罩和類別
        visualize.display_instances(
            original_image, 
            r['rois'], 
            r['masks'], 
            r['class_ids'], 
            self.val_data.class_names, 
            r['scores'], 
            ax=ax
        )

        plt.savefig(os.path.join(epoch_dir, f'epoch-{epoch + 1:02d}-objdet.png'))
        plt.close(fig)
        
        # 將檢測結果r傳遞給下面的mask函數
        self.mask_graph(epoch, epoch_dir, original_image, r)

    # 畫分割遮罩位置圖表
    def mask_graph(self, epoch, epoch_dir, original_image, detection_results):

        fig, ax = plt.subplots()
        visualize.display_top_masks(
            original_image, 
            detection_results['masks'], 
            detection_results['class_ids'], 
            self.val_data.class_names, 
            ax=ax
        )
        plt.savefig(os.path.join(epoch_dir, f'epoch-{epoch + 1:02d}-mask.png'))
        plt.close(fig)


    def on_epoch_end(self, epoch):
        # 創建當前 epoch 資料夾
        epoch_dir = os.path.join(self.save_dir, f'epoch-{epoch + 1:02d}')
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        # 呼叫生成損失圖的方法，將損失圖表存到當前 epoch 資料夾
        self.loss_graph(epoch, epoch_dir)
        
        # 呼叫可視化圖表方法（物件檢測、遮罩）
        self.objdet_graph(epoch, epoch_dir)
        
# 定義訓練類別
class Trainer:
    def __init__(self, model, dataset_train, dataset_val, config, model_dir):
        self.model = model
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.config = config
        self.model_dir = model_dir
        self.best_val_loss = float('inf')  # 設定一個初始的最佳驗證損失值(inf表示無窮大)
        self.patience = 5  # 設定容忍度為5次(驗證損失在5次epoch未改善停止訓練)
        self.wait = 0  # 設定等待次數為0

    def evaluate_validation_loss(self):  
        try:
            print("Evaluating validation loss...")
            val_loss = self.model.evaluate(self.dataset_val)
            print(f"Validation loss: {val_loss}")
            return val_loss
        except Exception as e:
            print(f"Error during validation loss evaluation: {e}")
            return float('inf')
    # def evaluate_validation_loss(self):  
    #     val_loss = self.model.evaluate(self.dataset_val)
    #     return val_loss 

    def train_single_epoch(self, learning_rate, layers):   
        try:
            print(f"Training with learning rate: {learning_rate}, layers: {layers}")
            self.model.train(self.dataset_train, self.dataset_val,
                             learning_rate=learning_rate,
                             epochs=1,  # 每次訓練一個 epoch
                             layers=layers)
            print("Single epoch training completed.")
        except Exception as e:
            print(f"Error during training single epoch: {e}")

    # def train_single_epoch(self, learning_rate, layers):    
    #     self.model.train(self.dataset_train, self.dataset_val,
    #                      learning_rate=learning_rate,
    #                      epochs=1,  # 每次訓練一個 epoch
    #                      layers=layers)
    

    def early_stopping_check(self, val_loss, epoch, layer_type):  
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
            print("Validation loss improved, saving model...")
            model_path = os.path.join(self.model_dir, f"best_model_{layer_type}_epoch_{epoch+1}.h5")
            try:
                self.model.keras_model.save_weights(model_path)
                print(f"Model weights saved at: {model_path}")
            except Exception as e:
                print(f"Error while saving model weights: {e}")
        else:
            self.wait += 1
            print(f"No improvement in validation loss. Wait count: {self.wait}/{self.patience}")
            if self.wait >= self.patience:
                print(f"Early stopping triggered for {layer_type} layers.")
                return True  # 返回 True 表示觸發早停
        return False
    # def early_stopping_check(self, val_loss, epoch, layer_type):  
    #     if val_loss < self.best_val_loss:
    #         self.best_val_loss = val_loss
    #         self.wait = 0
    #         print("Validation loss improved, saving model...")
    #         model_path = os.path.join(self.model_dir, f"best_model_{layer_type}_epoch_{epoch+1}.h5")
    #         self.model.keras_model.save_weights(model_path)
    #     else:
    #         self.wait += 1
    #         if self.wait >= self.patience:
    #             print(f"Early stopping triggered for {layer_type} layers.")
    #             return True  # 返回 True 表示觸發早停
    #     return False

    def train_head_layers(self, max_epochs=100): 
        for epoch in range(max_epochs):
            print(f"Training head layers - Epoch {epoch + 1}/{max_epochs}")
            try:

                self.train_single_epoch(learning_rate=self.config.LEARNING_RATE, layers='heads')
                
                # 計算驗證損失
                val_loss = self.evaluate_validation_loss()

                # 早停檢查
                if self.early_stopping_check(val_loss, epoch, layer_type='heads'):
                    break  
            except Exception as e:
                print(f"Error during head layer training in epoch {epoch + 1}: {e}")
                break
    # def train_head_layers(self, max_epochs=100): 

    #      for epoch in range(max_epochs):
    #         print(f"Training head layers - Epoch {epoch + 1}/{max_epochs}")
            
    #         # 單次訓練一個 epoch
    #         self.train_single_epoch(learning_rate=self.config.LEARNING_RATE, layers='heads')
            
    #         # 計算驗證損失
    #         val_loss = self.evaluate_validation_loss()

    #         # 早停檢查
    #         if self.early_stopping_check(val_loss, epoch, layer_type='heads'):
    #             break  

    def train_all_layers(self, max_epochs=50):
        self.wait = 0  # 重置等待計數器
        for epoch in range(max_epochs):
            print(f"Training all layers - Epoch {epoch + 1}/{max_epochs}")
            try:

                self.train_single_epoch(learning_rate=self.config.LEARNING_RATE / 10, layers='all')
                
                # 計算驗證損失
                val_loss = self.evaluate_validation_loss()

                # 早停檢查
                if self.early_stopping_check(val_loss, epoch, layer_type='all'):
                    break  # 如果早停被觸發，結束訓練
            except Exception as e:
                print(f"Error during all layers training in epoch {epoch + 1}: {e}")
                break
    #  定義訓練所有層(all)的函數。
    # def train_all_layers(self, max_epochs=50):
        
    #     self.wait = 0  # 重置等待計數器
        
    #     for epoch in range(max_epochs):
    #         print(f"Training all layers - Epoch {epoch + 1}/{max_epochs}")
            
    #         # 單次訓練一個 epoch
    #         self.train_single_epoch(learning_rate=self.config.LEARNING_RATE / 10, layers='all')
            
    #         # 計算驗證損失
    #         val_loss = self.evaluate_validation_loss()

    #         # 早停檢查
    #         if self.early_stopping_check(val_loss, epoch, layer_type='all'):
    #             break  # 如果早停被觸發，結束訓練


trainer = Trainer(model, dataset_train, dataset_val, config, MODEL_DIR)
trainer.train_head_layers()
trainer.train_all_layers()

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

"""## Detection"""

class InferenceConfig(ConnectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",config=inference_config,model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

"""## Evaluation"""

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))

