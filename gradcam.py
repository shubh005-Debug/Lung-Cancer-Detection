import numpy as np
import cv2
import base64
import logging
import io

logger = logging.getLogger(__name__)


def generate_gradcam(model, img_array, pred_index):
    """
    Generate Grad-CAM heatmap for the predicted class.
    Works with VGG16-based Sequential model.
    Returns base64 encoded heatmap overlay image.
    """
    try:
        import tensorflow as tf

        # Find last Conv2D layer in the model
        last_conv_layer = None
        for layer in reversed(model.layers):
            # Handle nested model (VGG16 inside Sequential)
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sublayer.name
                        last_conv_layer = sublayer
                        break
            if last_conv_layer:
                break

        if last_conv_layer is None:
            logger.warning("No Conv2D layer found for Grad-CAM")
            return None

        # Build grad model: inputs → [last_conv_output, final_output]
        inner_model = model.layers[0]  # VGG16
        grad_model = tf.keras.models.Model(
            inputs=inner_model.input,
            outputs=[last_conv_layer.output, inner_model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        # Resize heatmap to 224x224
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Overlay on original image
        original = np.uint8(img_array[0] * 255)
        overlay = cv2.addWeighted(original, 0.55, heatmap_colored, 0.45, 0)

        # Encode to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        encoded = base64.b64encode(buffer).decode('utf-8')

        logger.info("✅ Grad-CAM generated successfully.")
        return encoded

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}")
        return None