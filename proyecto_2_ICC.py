# Proyecto: Clasificador de Dígitos Escritos a Mano
# Curso: Introducción a la Ciencia de la Computación

import cv2
import numpy as np
import matplotlib.pyplot as plt
# LÍNEA MÁGICA PARA ARREGLAR EL ERROR:
plt.switch_backend('TkAgg')
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ============================================================================
# PASO 1: CARGAR Y PREPARAR LOS DATOS
# ============================================================================

print("Cargando el dataset de dígitos...")
digits = load_digits()
imagenes = digits.images  # Imágenes de 8x8 píxeles
etiquetas = digits.target  # Números del 0 al 9

print(f"Total de imágenes: {len(imagenes)}")
print(f"Tamaño de cada imagen: {imagenes[0].shape}")
print(f"Dígitos disponibles: {set(etiquetas)}")

# ============================================================================
# PASO 2: CALCULAR EL PROMEDIO DE CADA DÍGITO
# ============================================================================

print("\nCalculando el promedio de cada dígito...")
promedios = []

for digito in range(10):  # Del 0 al 9
    # Encontrar todas las imágenes de este dígito
    imagenes_del_digito = []
    for i in range(len(imagenes)):
        if etiquetas[i] == digito:
            imagenes_del_digito.append(imagenes[i])

    # Calcular el promedio
    imagenes_del_digito = np.array(imagenes_del_digito)
    promedio = np.mean(imagenes_del_digito, axis=0)
    promedios.append(promedio)

promedios = np.array(promedios)
print("¡Promedios calculados!")


# ============================================================================
# PASO 3: MOSTRAR LOS PROMEDIOS (OPCIONAL)
# ============================================================================

def mostrar_imagen(imagen, titulo=""):
    """Función simple para mostrar una imagen"""
    plt.figure(figsize=(3, 3))
    plt.imshow(imagen, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()


# Mostrar los promedios de cada dígito
respuesta = input("\n¿Quieres ver los promedios de cada dígito? (s/n): ")
if respuesta.lower() == 's':
    for i in range(10):
        mostrar_imagen(promedios[i], f'Promedio del dígito {i}')

# ============================================================================
# PASO 4: DIVIDIR LOS DATOS EN ENTRENAMIENTO Y PRUEBA
# ============================================================================

print("\nDividiendo los datos en entrenamiento (80%) y prueba (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    imagenes, etiquetas, test_size=0.2, random_state=42
)

print(f"Imágenes de entrenamiento: {len(X_train)}")
print(f"Imágenes de prueba: {len(X_test)}")


# ============================================================================
# PASO 5: MÉTODO 1 - CLASIFICACIÓN POR VECINOS MÁS CERCANOS
# ============================================================================

def calcular_distancia(imagen1, imagen2):
    """Calcula la distancia entre dos imágenes"""
    # Convertir a vectores planos
    vec1 = imagen1.flatten()
    vec2 = imagen2.flatten()

    # Calcular distancia euclidiana
    distancia = np.sqrt(np.sum((vec1 - vec2) ** 2))
    return distancia


def clasificar_metodo1(imagen_nueva, imagenes_entrenamiento, etiquetas_entrenamiento, k=3):
    """Clasifica una imagen usando los k vecinos más cercanos"""
    distancias = []

    # Calcular distancia a todas las imágenes de entrenamiento
    for i in range(len(imagenes_entrenamiento)):
        dist = calcular_distancia(imagen_nueva, imagenes_entrenamiento[i])
        distancias.append((dist, etiquetas_entrenamiento[i]))

    # Ordenar por distancia (menor a mayor)
    distancias.sort()

    # Tomar los k más cercanos
    vecinos_cercanos = distancias[:k]
    etiquetas_cercanas = [etiqueta for _, etiqueta in vecinos_cercanos]

    # Votar por mayoría
    votos = {}
    for etiqueta in etiquetas_cercanas:
        if etiqueta in votos:
            votos[etiqueta] += 1
        else:
            votos[etiqueta] = 1

    # Encontrar la etiqueta con más votos
    mejor_etiqueta = max(votos, key=votos.get)
    return mejor_etiqueta, etiquetas_cercanas


# ============================================================================
# PASO 6: MÉTODO 2 - CLASIFICACIÓN POR DISTANCIA A PROMEDIOS
# ============================================================================

def clasificar_metodo2(imagen_nueva, promedios):
    """Clasifica una imagen comparándola con los promedios de cada dígito"""
    distancias_a_promedios = []

    for digito in range(10):
        distancia = calcular_distancia(imagen_nueva, promedios[digito])
        distancias_a_promedios.append(distancia)

    # El dígito predicho es el que tiene menor distancia
    digito_predicho = distancias_a_promedios.index(min(distancias_a_promedios))
    return digito_predicho


# ============================================================================
# PASO 7: EVALUAR AMBOS MÉTODOS
# ============================================================================

print("\nEvaluando ambos métodos...")
print("Esto puede tomar unos segundos...")

# Método 1: Vecinos más cercanos
aciertos_metodo1 = 0
predicciones_metodo1 = []

for i in range(len(X_test)):
    prediccion, _ = clasificar_metodo1(X_test[i], X_train, y_train)
    predicciones_metodo1.append(prediccion)
    if prediccion == y_test[i]:
        aciertos_metodo1 += 1

precision_metodo1 = aciertos_metodo1 / len(X_test)

# Método 2: Distancia a promedios
aciertos_metodo2 = 0
predicciones_metodo2 = []

for i in range(len(X_test)):
    prediccion = clasificar_metodo2(X_test[i], promedios)
    predicciones_metodo2.append(prediccion)
    if prediccion == y_test[i]:
        aciertos_metodo2 += 1

precision_metodo2 = aciertos_metodo2 / len(X_test)

# ============================================================================
# PASO 8: MOSTRAR RESULTADOS
# ============================================================================

print("\n" + "=" * 50)
print("RESULTADOS DE LA EVALUACIÓN")
print("=" * 50)
print(f"Método 1 (Vecinos cercanos): {precision_metodo1:.2%} de precisión")
print(f"Método 2 (Distancia a promedios): {precision_metodo2:.2%} de precisión")

# Contar aciertos por cada dígito
print("\nDetalle por dígito:")
for digito in range(10):
    # Contar cuántas veces aparece este dígito en las pruebas
    total_digito = 0
    aciertos_m1 = 0
    aciertos_m2 = 0

    for i in range(len(y_test)):
        if y_test[i] == digito:
            total_digito += 1
            if predicciones_metodo1[i] == digito:
                aciertos_m1 += 1
            if predicciones_metodo2[i] == digito:
                aciertos_m2 += 1

    if total_digito > 0:
        print(f"Dígito {digito}: Método 1 = {aciertos_m1}/{total_digito} ({aciertos_m1 / total_digito:.1%}), "
              f"Método 2 = {aciertos_m2}/{total_digito} ({aciertos_m2 / total_digito:.1%})")


# ============================================================================
# PASO 9: PROBAR CON UNA IMAGEN NUEVA
# ============================================================================

def leer_imagen_externa(ruta_archivo):
    """Lee una imagen externa y la prepara para clasificación"""
    try:
        # Leer la imagen en escala de grises
        imagen = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f"No se pudo cargar la imagen desde {ruta_archivo}")
            return None

        # Redimensionar a 8x8 píxeles
        imagen_pequena = cv2.resize(imagen, (8, 8))

        # Invertir colores (fondo blanco, números negros -> fondo negro, números blancos)
        imagen_invertida = 255 - imagen_pequena

        # Escalar valores de 0-255 a 0-16 (como el dataset original)
        imagen_final = (imagen_invertida / 255.0) * 16

        return imagen_final

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None


print("\n" + "=" * 50)
print("PRUEBA CON UNA IMAGEN")
print("=" * 50)
print("Opciones:")
print("1. Usar imagen externa ('image-test.jpeg')")
print("2. Usar una imagen del conjunto de prueba")

opcion = input("Elige una opción (1 o 2): ").strip()

if opcion == "1":
    imagen_prueba = leer_imagen_externa("image-test.jpeg")
    if imagen_prueba is not None:
        print("Imagen externa cargada exitosamente")
        mostrar_imagen(imagen_prueba, "Imagen cargada desde archivo")
    else:
        print("No se pudo cargar la imagen externa. Usando imagen de prueba...")
        imagen_prueba = X_test[0]
        print(f"Etiqueta real: {y_test[0]}")
        mostrar_imagen(imagen_prueba, f"Imagen de prueba (etiqueta real: {y_test[0]})")
else:
    imagen_prueba = X_test[0]
    print(f"Etiqueta real: {y_test[0]}")
    mostrar_imagen(imagen_prueba, f"Imagen de prueba (etiqueta real: {y_test[0]})")

# Clasificar con ambos métodos
resultado_m1, vecinos = clasificar_metodo1(imagen_prueba, X_train, y_train)
resultado_m2 = clasificar_metodo2(imagen_prueba, promedios)

print(f"\nResultados de la clasificación:")
print(f"Método 1 (Vecinos cercanos): El dígito es {resultado_m1}")
print(f"  - Los 3 vecinos más cercanos fueron: {vecinos}")
print(f"Método 2 (Distancia a promedios): El dígito es {resultado_m2}")

print("\n¡Proyecto completado!")