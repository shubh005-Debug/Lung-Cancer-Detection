import logging
import os

logger = logging.getLogger(__name__)

MODEL_PATH = r"C:\Users\SANIDHYA\OneDrive\Attachments\Desktop\lung-cancer-app\backend\train_lung_cancer_model.keras"
WEIGHTS_PATH = r"C:\Users\SANIDHYA\OneDrive\Attachments\Desktop\lung-cancer-app\backend\bests_model.weights.h5"


class ModelLoader:
    """Loads VGG16-based lung cancer CNN model."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or self._detect_model_path()
        self._model = None
        self._loaded = False
        logger.info(f"Model file detected: {self.model_path}")

    def _detect_model_path(self):
        # Check hardcoded path first
        if os.path.exists(MODEL_PATH):
            return MODEL_PATH
        # Fallback: scan backend folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(base_dir):
            if f.endswith(".keras") or f.endswith(".h5"):
                return os.path.join(base_dir, f)
        return MODEL_PATH

    def load(self):
        try:
            import tensorflow as tf

            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}. Using mock model.")
                self._model = self._create_mock_model()
                self._loaded = True
                return

            # Strategy 1: Direct load (standard)
            try:
                self._model = tf.keras.models.load_model(self.model_path)
                logger.info("✅ Model loaded successfully (standard).")
                self._loaded = True
                return
            except Exception as e1:
                logger.warning(f"Strategy 1 failed: {e1}")

            # Strategy 2: compile=False
            try:
                self._model = tf.keras.models.load_model(self.model_path, compile=False)
                self._model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                logger.info("✅ Model loaded successfully (compile=False).")
                self._loaded = True
                return
            except Exception as e2:
                logger.warning(f"Strategy 2 failed: {e2}")

            # Strategy 3: Rebuild VGG16 architecture + load weights from .keras
            try:
                self._model = self._rebuild_and_load_weights(self.model_path)
                logger.info("✅ Model loaded successfully (rebuild + load weights).")
                self._loaded = True
                return
            except Exception as e3:
                logger.warning(f"Strategy 3 failed: {e3}")

            # Strategy 4: Rebuild + load from bests_model.weights.h5 if exists
            try:
                base_dir = os.path.dirname(os.path.abspath(self.model_path))
                weights_file = os.path.join(base_dir, "bests_model.weights.h5")
                if os.path.exists(weights_file):
                    self._model = self._build_vgg16_model()
                    self._model.load_weights(weights_file)
                    logger.info("✅ Model loaded from bests_model.weights.h5.")
                    self._loaded = True
                    return
                else:
                    raise FileNotFoundError("bests_model.weights.h5 not found")
            except Exception as e4:
                logger.warning(f"Strategy 4 failed: {e4}")

            # Final fallback: mock model
            logger.error("All strategies failed. Using mock model.")
            self._model = self._create_mock_model()
            self._loaded = True

        except ImportError:
            logger.warning("TensorFlow not installed. Using mock model.")
            self._model = None
            self._loaded = True

    def _build_vgg16_model(self):
        """Rebuild exact architecture from training notebook."""
        import tensorflow as tf

        OUTPUT_SIZE = 4

        pretrained_model = tf.keras.applications.VGG16(
            weights=None,           # No imagenet weights needed — loading saved weights
            include_top=False,
            input_shape=(224, 224, 3)
        )

        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _rebuild_and_load_weights(self, keras_path):
        """Build VGG16 model and load weights from .keras file."""
        import tensorflow as tf

        model = self._build_vgg16_model()

        # Build model with dummy forward pass so weights can be loaded
        import numpy as np
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)

        # Load weights only from the .keras file
        model.load_weights(keras_path)
        return model

    def _create_mock_model(self):
        """Minimal mock model for demo purposes."""
        try:
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(4, activation="softmax")
            ])
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            logger.info("⚠️ Demo mock model active. Replace with trained model for real predictions.")
            return model
        except Exception:
            return None

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model(self):
        return self._model