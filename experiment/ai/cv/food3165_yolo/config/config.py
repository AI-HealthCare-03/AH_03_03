# no package for independent configuration management

# Initialize the parameters
CONF_THRES = 0.1 #0.5  # Confidence threshold
NMS_THRES  = 0.1 #0.4  # Non-maximum suppression threshold

INPWIDTH  = 416 # 32*10  # 608     #Width of network's input image # 320(32*10)
INPHEIGHT = 416 #32*9 # 608     #Height of network's input image # 288(32*9) best

# start video frame number
Video_Start_Frame = (21-4)*60*30+(2-39)*30 # compute the first starting location from a video

# model base dir
# ModelBaseDir = "C:/Users/mmc/workspace/AI_core/food3165_yolo/yolo"
ModelBaseDir = "./yolo"
TEST_IMAGE_PATH = "./images_rec/13118.jpg"
# TEST_IMAGE_PATH = "C:/Users/mmc/workspace/AI_core/food3165_yolo/yolo/data/쑥개떡/A240213_111121_0005.jpg"
# TEST_IMAGE_PATH = "./yolo/data/food/images/A270309_111112_0002.jpg"
TEST_VIDEO_PATH = \
    ""#"E:/Topes_data_related/시나리오 영상/시나리오 영상/20200909PM/6085-20200909-170439-1599638679.mp4"
SHOW_TEXT_FLAG = 1
PS_FLAG = 1

# Load names of classes, please don't include the first directory separator like "/data/..."
CLASSES_FILE = "data/food/food3165-classes.names"
CLASSES_FILE_CODE = "data/food/food3165-classes.codes"

# Give the configuration and weight files for the model and load the network using them.
# Don't include the first directory separator
# -- yolov3 ------
# ------- 3 layers
# itms
# Model_Configuration = "config/itms-dark-yolov3-tiny_3l-v3-2.cfg"
# Model_Weights = "data/food/weights/itms-dark-yolov3-tiny_3l-v3-2_200000.weights"
# food
Model_Configuration = "config/food3165-yolov3-tiny_3l-v3-2.cfg"
Model_Weights = "data/food/weights/food-dark-yolov3-tiny_3l-v3-2_24000.weights"
# ------- full layers
#Model_Configuration = "config/food-dark-yolov3-full-2.cfg"
# Model_Weights = "data/food/weights/food-dark-yolov3-full-2_100000.weights"
# -- yolov4 -------
# 3l layers
# Model_Configuration = "config/food-dark-yolov4-tiny-3l-v1.cfg"
# Model_Weights       = "data/food/weights/food-dark-yolov4-tiny-3l-v1_best.weights"
# full layers
# Model_Configuration = "config/food-dark-yolov4-full.cfg"
# Model_Weights       = "data/food/weights/food-dark-yolov4-full_10000.weights"

