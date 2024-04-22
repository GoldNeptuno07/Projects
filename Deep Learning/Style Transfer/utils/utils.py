# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 17:51:25 2024

@author: Angel Cruz
"""

"""

    Script with several auxiliary functions, such as, load an image, transform a
    tensor to an image, calculate the gram matrix given a feature map, return the
    model that will give us the feature maps, a class to return the features of 
    the content and style layers and finally a function to compute the loss.
    
"""

import PIL.Image
import tensorflow as tf
import numpy as np


# =============================================================================
# Load Image
# =============================================================================
def load_img(path_to_img, limit_pixels_to= 224):
    """
        Function to load an image with tensorflow given the image path. 
        The function will limit the width or height based on limit_pixels_to.
    """
    
    max_dim= limit_pixels_to
    # Load Image
    img= tf.io.read_file(path_to_img)
    img= tf.image.decode_image(img, channels= 3)
    img= tf.image.convert_image_dtype(img, tf.float32) # img to tensor
    # Resize Image
    shape= tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim= max(shape)
    scale= max_dim / long_dim
    new_shape= tf.cast(shape * scale, tf.int32)
    img= tf.image.resize(img, new_shape)
    img= img[tf.newaxis, :] # Add a new dimension at idx 0
    return img

# =============================================================================
# Tensor to Image
# =============================================================================

def tensor_to_image(tensor):
    """
        Function to convert a tensor to an image using PIL.Image. 
        The function supposes that the tensor goes from [0.0, 1.0].
    """
    
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# =============================================================================
# VGG Layers
# =============================================================================
def vgg_layers(layers_name):
    """
        Function to load the VGG19 model and personalize the intermediate outputs
        we want, based on the style_layers and content_layers. Finally, the model
        will be return.
    """
    
    # Load the Model
    vgg= tf.keras.applications.VGG19(include_top=False, weights= 'imagenet')
    vgg.trainable= False
    # Get the VGG layers based on layers_name
    outputs= [vgg.get_layer(name).output for name in layers_name]
    # Model to get intermediate outputs
    model= tf.keras.Model([vgg.input], outputs)
    return model


# =============================================================================
# Gram Matrix
# =============================================================================
def gram_matrix(input_tensor):
    """
        Function to compute the gram matrix given a feature map. The output
        will consist of a matrix of [1, channels, channels].
    """
    result= tf.linalg.einsum('bijc, bijd->bcd', input_tensor, input_tensor)
    input_shape= tf.shape(input_tensor)
    num_locations= tf.cast(input_shape[1]*input_shape[2], dtype= tf.float32)
    return result / num_locations

# =============================================================================
# Style Content Model
# =============================================================================
class StyleContentModel(tf.keras.models.Model):
    """
        Class to return the content and style features, the style features 
        will consist of the gram matrix of each style layer. The model will
        return a dictionary with two main keys, content and style.
    """
    def __init__(self, style_layers, content_layers):
        """
            style_layers. A str-list with the names of the style layers, for example:
                            ['block1_conv1','block2_conv1',...]
            
            content_layers. A str-list with the names of the content layers, for example:
                            ['block1_conv1','block2_conv1',...]
        """
        super(StyleContentModel, self).__init__()
        self.vgg= vgg_layers(style_layers + content_layers)
        self.style_layers= style_layers
        self.content_layers=content_layers
        self.num_style_layers= len(style_layers)
        self.vgg.trainable= False
    
    def call(self, inputs):
        inputs= inputs*255.0
        preprocessed_input= tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs= self.vgg(preprocessed_input)
        style_outputs, content_outputs= (outputs[:self.num_style_layers],
                                        outputs[self.num_style_layers:])
        style_gram_matrixs= [gram_matrix(feature_map) for feature_map in style_outputs]
        
        content_dict= {content_name: value 
                       for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict= {style_name: value 
                       for style_name, value in zip(self.style_layers, style_gram_matrixs)}
        return {'content': content_dict, 'style': style_dict}


# =============================================================================
# Style & Content Loss
# =============================================================================
def style_content_loss(features, style_target, 
                       content_target, style_weight= 0.01, content_weight= 1000):
    """
        This function computes the overall loss, based on the content loss and style loss.
            Parameters:
                features. A dictionary containing the style and content features.
                style_target. A dictionary whose keys are the style layers and the values are 
                            the gram arrays.
                content_target. A dictionary whose keys are the content layers and the values are 
                                the feature maps.
                style_weight. A float value, default 0.01. Represent the importance of the style.
                content_weight. A float value, default 100. Represent the importance of the content.
    """
    
    # Split the style and content feature maps
    style_outputs= features['style'] 
    content_outputs= features['content']
    
    style_loss= tf.add_n([tf.reduce_mean((style_outputs[name]-style_target[name])**2) 
                         for name in style_outputs.keys()])
    style_loss *=  style_weight / len(style_outputs)
    
    content_loss= tf.add_n([tf.reduce_mean((content_outputs[name]-content_target[name])**2) 
                         for name in content_outputs.keys()])
    content_loss *=  content_weight / len(content_outputs)
    
    loss= style_loss + content_loss
    return loss






