# ExamenTratDatosUIDE23

Consideraciones:
- Es imprescindible descomprimir el siguiente dataset en la carpeta '/CarneDataset' para poder usar las imagenes y entrenar el modelo:
  https://drive.google.com/file/d/1Z5DJ-MVS1TQV1kow9mIFWTec-ZdOLRLF/view?usp=sharing

Para este ejercicio se usaron redes neuronales convolucionales, pues estas utilizan datos tridimensionales para tareas de clasificación de imágenes y nos permiten lograr un reconocimiento acertado de objetos.

Se uso la propiedad de TF, image_size=(300,300) para estandarizar el tamano de todas las imagenes de nuestro dataset y que esto beneficie al entrenamiento de la RN.
Se uso el modelo secuencial de TF como se puede ver a continuacion:

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(300, 300, 3)),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(), tf.keras.layers.Dropout(0.6),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

Se puede observar que nuestro modelo tiene 10 capas, la capa 1 corresponde a la normalización, la segunda es una capa de convolución con 16 filtros, la tercera es una capa de max pooling, la cuarta es una capa de convolución con 32 filtros, la quinta es una capa de max pooling, la sexta es una capa de convolución con 64 filtros, la séptima es una capa de max pooling, la octava es una capa flatten con un Dropout de 0.6 para normalizar nuestra red, la novena es una capa densa con 128 neuronas y la última es una capa densa con 8 neuronas, lo que corresponde a las 8 clases de carnes en nuestro dataset.

Esta es la estructura básica de cualquier red CNN, la cual se compone de capas de convolución, capas de max pooling y finalmente capas densas.

Ahora compilamos el modelo.

Usamos el optimizador Adam para compilar nuestro modelo ya que este modelo es computacionalmente eficiente, requiere poca memoria, es invariable al cambio de escala diagonal de gradientes y es adecuado para problemas que son grandes en términos de datos/parámetros".

No se modifico el aspecto de las imagenes en cuanto al color ya que al ser imagenes sin mucho detalle era importante hacer uso de los colores y contrastes entre rojos para el entrenamiento y clasificacion de las carnes.



