import subprocess
import pandas as pd
import cv2 
import numpy as np
import os
import wget
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt
from object_detection.utils import config_util
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2



WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'

# Definir etiquetas
labels = [
    {'name': 'comida', 'id': 1},
    {'name': 'esposo', 'id': 2},
    {'name': 'te_amo', 'id': 3},
    {'name': 'hola', 'id': 4}
]

# Crear carpeta de anotaciones si no existe
if not os.path.exists(ANNOTATION_PATH):
    os.makedirs(ANNOTATION_PATH)

# Generar archivo label_map.pbtxt
with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# Descargar el modelo preentrenado
url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
filename = wget.download(url, out=PRETRAINED_MODEL_PATH)

# Mover y extraer el archivo descargado
tar_path = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz')
subprocess.run(['tar', '-zxvf', tar_path, '-C', PRETRAINED_MODEL_PATH])

# Crear carpeta del modelo personalizado
model_dir = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Copiar el archivo pipeline.config al directorio del modelo
config_source_path = os.path.join(PRETRAINED_MODEL_PATH, 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8', 'pipeline.config')
config_dest_path = os.path.join(model_dir, 'pipeline.config')
os.system(f'cp {config_source_path} {config_dest_path}')

# Ejecutar el comando para generar el archivo train.record
subprocess.run(['python', SCRIPTS_PATH + '/generate_tfrecord.py',
                '-x', IMAGE_PATH + '/train',
                '-l', ANNOTATION_PATH + '/label_map.pbtxt',
                '-o', ANNOTATION_PATH + '/train.record'])

# Ejecutar el comando para generar el archivo test.record
subprocess.run(['python', SCRIPTS_PATH + '/generate_tfrecord.py',
                '-x', IMAGE_PATH + '/test',
                '-l', ANNOTATION_PATH + '/label_map.pbtxt',
                '-o', ANNOTATION_PATH + '/test.record'])



config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
# print(config)

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  
    
pipeline_config.model.ssd.num_classes = 5
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   
    
print("""python {}/research/object_detection/model_main_tf2.py --model_dir={}/{} --pipeline_config_path={}/{}/pipeline.config --num_train_steps=5000""".format(APIMODEL_PATH, MODEL_PATH,CUSTOM_MODEL_NAME,MODEL_PATH,CUSTOM_MODEL_NAME))
