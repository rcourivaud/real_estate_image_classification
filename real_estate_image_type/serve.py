import torch
import json
import time

import logging, requests, os, io
from fastai.vision import open_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    if content_type == JPEG_CONTENT_TYPE: return open_image(io.BytesIO(request_body))
    # process a URL submitted to the endpoint
    if content_type == JSON_CONTENT_TYPE:
        img_request = requests.get(request_body['url'], stream=True)
        return open_image(io.BytesIO(img_request.content))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def model_fn(model_dir):
    logger.info('model_fn')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model = torch.load(f)

    return model.to(device)


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    logger.info("--- Inference time: %s seconds ---" % (time.time() - start_time))
    logger.info(f'Predicted class is {str(predict_class)}')
    logger.info(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    return {"class":str(predict_class),
            "confidence":predict_values[predict_idx.item()].item()}

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))