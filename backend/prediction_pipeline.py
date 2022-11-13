import tensorflow as tf
import numpy as np


def predict(image, model_path='model.tflite', confidence_threshold=0.11):
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)
    
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.image.resize_with_pad(image, 256, 256)
    
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())

    interpreter.invoke()

    # Output is a [1, 1, 17, 3] numpy array.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    filtered_keypoints = []
    
    for row in keypoints_with_scores[0, 0, :, :]:
        # if greater than confidence threshold
        if row[2] > confidence_threshold:
            filtered_keypoints.append(row)
            
    filtered_keypoints = np.array(filtered_keypoints)[..., :2]
    
    temp = filtered_keypoints[:, 0].copy()
    filtered_keypoints[:, 0] = filtered_keypoints[:, 1]
    filtered_keypoints[:, 1] = temp
    
    return filtered_keypoints

def calc_distance(x, y):
    return np.sqrt(np.sum(np.square(x-y)))