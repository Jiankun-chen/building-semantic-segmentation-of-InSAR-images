import tensorflow as tf


def add_output_images(images, logits, labels):
    cast_labels = tf.cast(labels, tf.uint8)
    cast_labels = tf.cast(cast_labels[...,None], tf.float32) * 128
    tf.summary.image('input_labels', cast_labels, max_outputs=20)
    classification0 = tf.nn.softmax(logits=logits, dim=-1)[..., 0:1]
    classification1 = tf.nn.softmax(logits=logits, dim=-1)[..., 1:2]
    classification2 = tf.nn.softmax(logits=logits, dim=-1)[..., 2:3]
    output_image_gb = images[..., 0]
    output_image_r = (1-classification2) + tf.multiply(images[..., 0:1], classification2)
    output_image_g = (1-classification1) + tf.multiply(images[..., 0:1], classification1)
    output_image_b = (1-classification0) + tf.multiply(tf.cast(128, tf.float32), classification0)
    output_image = tf.stack([0.5*(output_image_r+output_image_g), 0.5*output_image_g, 0.5*(output_image_b+output_image_g)], axis=3)
    output_image = tf.squeeze(output_image)
    tf.summary.image('output_mixed', output_image, max_outputs=20)
    output_image_binary = tf.argmax(logits, 3)
    output_image_binary = tf.cast(output_image_binary[...,None], tf.float32) * 128
    tf.summary.image('output_labels', output_image_binary, max_outputs=20)

    return

