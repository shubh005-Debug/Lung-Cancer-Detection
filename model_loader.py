
import logging
import os

import gdown
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env for local development
# Render environment variables are loaded automatically
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "train_lung_cancer_model.keras")
WEIGHTS_PATH = os.path.join(BASE_DIR, "bests_model.weights.h5")

MODEL_GDRIVE_ID = os.getenv("MODEL_GDRIVE_ID")


def download_model():
    """Download trained model from Google Drive."""

    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        logger.info("Model file already exists.")
        return

    model_id = os.getenv("MODEL_GDRIVE_ID")

    if not model_id:
        raise ValueError(
            "MODEL_GDRIVE_ID is missing from environment variables."
        )

    logger.info("Downloading trained model from Google Drive...")

    url = f"https://drive.google.com/uc?id={model_id}"

    downloaded_file = gdown.download(
        url=url,
        output=MODEL_PATH,
        quiet=False
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
        self._is_real_model = False

        logger.info(f"Model path: {self.model_path}")

    def load(self):
        try:
            import tensorflow as tf

            # Download trained model if missing
            if not os.path.exists(self.model_path):
                if self.model_path == MODEL_PATH:
                    download_model()
                else:
                    raise FileNotFoundError(self.model_path)

            # Check if file is empty
            if os.path.getsize(self.model_path) == 0:
                raise RuntimeError("Model file is empty.")

            # Strategy 1: Standard loading
            try:
                self._model = tf.keras.models.load_model(
                    self.model_path
                )

                logger.info("Trained model loaded successfully.")
                self._loaded = True
                self._is_real_model = True
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

                logger.info(
                    "Trained model loaded successfully (compile=False)."
                )

                self._loaded = True
                self._is_real_model = True
                return

            except Exception as e2:
                logger.error(
                    f"compile=False loading failed: {e2}"
                )

            # Strategy 3: Load separate weights if available
            try:
                if os.path.exists(WEIGHTS_PATH):
                    self._model = self._build_vgg16_model()
                    self._model.load_weights(WEIGHTS_PATH)

                    logger.info(
                        "Trained model loaded from weights file."
                    )

                    self._loaded = True
                    self._is_real_model = True
                    return

                raise FileNotFoundError(
                    "Separate weights file not found."
                )

            except Exception as e3:
                logger.error(
                    f"Weights loading failed: {e3}"
                )

            # Fallback if trained model cannot be loaded
            logger.error(
                "Real model loading failed. "
                "Activating demo mock model."
            )

            self._model = self._create_mock_model()
            self._loaded = self._model is not None
            self._is_real_model = False

        except Exception as e:
            logger.exception(
                f"Model initialization failed: {e}"
            )

            self._model = self._create_mock_model()
            self._loaded = self._model is not None
            self._is_real_model = False

    def _build_vgg16_model(self):
        """Rebuild VGG16 model architecture for weights loading."""

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
            tf.keras.layers.Dense(
                OUTPUT_SIZE,
                activation="softmax"
            )
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
        """Creates a demo model. Not a trained medical model."""

        try:
            import tensorflow as tf

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(
                    4,
                    activation="softmax"
                )
            ])

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            logger.warning(
                "DEMO MOCK MODEL ACTIVE. "
                "This is NOT a trained medical prediction model."
            )

            return model

        except Exception:
            logger.exception("Mock model creation failed.")
            return None

    def is_loaded(self):
        return self._loaded

    def is_real_model(self):
        return self._is_real_model

    def get_model(self):
        return self._model
