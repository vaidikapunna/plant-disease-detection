from tensorflow.keras.models import load_model

model = load_model("plant_disease_model.h5")
model.summary()
