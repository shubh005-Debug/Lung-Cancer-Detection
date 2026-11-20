import logging
import os

import gdown
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env locally; Render environment variables are also supported
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "train_lung_cancer_model.keras")
WEIGHTS_PATH = os.path.join(BASE_DIR, "bests_model.weights.h5")

MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID")


def download_model():
    """Download trained model from Google Drive if it doesn't exist."""

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        logger.info("Model file already exists.")
        return

    if not MODEL_GDRIVE_ID:
        raise ValueError("MODEL_GDRIVE_ID is missing from environment variables.")

    logger.info("Downloading trained model from Google Drive...")

    url = f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}"

    downloaded_file = gdown.download(
        url=url,
        output=MODEL_PATH,
        quiet=False,
        fuzzy=True
    )

    if not downloaded_file or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Google Drive model download failed.")

    if os.path.getsize(MODEL_PATH) == 0:
        raise RuntimeError("Downloaded model file is empty.")

    logger.info("Model downloaded successfully.")


class ModelLoader:
    """Loads VGG16-based lung cancer CNN model."""

    def __init__(self, model_path=None):
        self.model_path = model_path or MODEL_PATH
        self._model = None
        self._loaded = False

        logger.info(f"Model path: {self.model_path}")

    def load(self):
        try:
            import tensorflow as tf

            # Download model if it is missing
            if not os.path.exists(self.model_path):
                if self.model_path == MODEL_PATH:
                    download_model()
                else:
                    raise FileNotFoundError(self.model_path)

            # Strategy 1: Standard model loading
            try:
                self._model = tf.keras.models.load_model(self.model_path)

                logger.info("Model loaded successfully (standard).")
                self._loaded = True
                return

            except Exception as e1:
                logger.warning(f"Standard loading failed: {e1}")

            # Strategy 2: Load without compilation
            try:
                self._model = tf.keras.models.load_model(
                    self.model_path,
                    compile=False
                )

                self._model.compile(
                    optimizer="adam",
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

                logger.info("Model loaded successfully (compile=False).")
                self._loaded = True
                return

            except Exception as e2:
                logger.error(f"compile=False loading failed: {e2}")

            # Strategy 3: Load separately saved weights, if available
            try:
                if os.path.exists(WEIGHTS_PATH):
                    self._model = self._build_vgg16_model()
                    self._model.load_weights(WEIGHTS_PATH)

                    logger.info("Model loaded from separate weights file.")
                    self._loaded = True
                    return

                raise FileNotFoundError(
                    "Separate weights file not found."
                )

            except Exception as e3:
                logger.error(f"Weights loading failed: {e3}")

            # Final fallback
            logger.error(
                "All real model loading strategies failed. "
                "Demo mock model will be used."
            )

            self._model = self._create_mock_model()
            self._loaded = True

        except Exception as e:
            logger.exception(f"Model initialization failed: {e}")

            self._model = self._create_mock_model()
            self._loaded = True

    def _build_vgg16_model(self):
        """Rebuild VGG16 architecture."""

        import tensorflow as tf

        OUTPUT_SIZE = 4

        pretrained_model = tf.keras.applications.VGG16(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        )

        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(OUTPUT_SIZE, activation="softmax")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.0001
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def _create_mock_model(self):
        """Minimal mock model for demo purposes only."""

        try:
            import tensorflow as tf

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(4, activation="softmax")
            ])

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            logger.warning(
                "Demo mock model active. "
                "It is NOT a trained medical prediction model."
            )

            return model

        except Exception:
            return None

    def is_loaded(self):
        return self._loaded

    def get_model(self):
        return self._model
