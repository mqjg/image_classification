import boto3
import tensorflow as tf

# Create a low-level client representing Amazon SageMaker Runtime
sagemaker_runtime = boto3.client(
    "sagemaker-runtime", region_name='us-east-2')

# The endpoint name must be unique within 
# an AWS Region in your AWS account. 
endpoint_name='testModel-Endpoint-20231213-180912'

sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img_height = 180
img_width = 180
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# Gets inference from the model hosted at the specified endpoint:
response = sagemaker_runtime.invoke_endpoint(
    EndpointName=endpoint_name, 
    Body=bytes(img_array, 'utf-8')
    )

# Decodes and prints the response body:
print(response['Body'].read().decode('utf-8'))