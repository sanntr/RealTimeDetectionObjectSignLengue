import os

def renombrar_archivos_incrementales(carpeta, base_nombre):
    try:
        # Obtener la lista de archivos en la carpeta
        archivos = os.listdir(carpeta)
        contador = 1

        for archivo in archivos:
            # Obtener la ruta completa del archivo
            ruta_completa = os.path.join(carpeta, archivo)

            # Verificar que sea un archivo
            if os.path.isfile(ruta_completa):
                # Obtener la extensión del archivo
                _, extension = os.path.splitext(archivo)

                # Crear el nuevo nombre con el formato deseado
                nuevo_nombre = f"{base_nombre}_{contador}{extension}"

                # Ruta completa del nuevo archivo
                nueva_ruta = os.path.join(carpeta, nuevo_nombre)

                # Renombrar el archivo
                os.rename(ruta_completa, nueva_ruta)
                print(f"Renombrado: {archivo} -> {nuevo_nombre}")
                contador += 1

        print("Renombrado completado.")
    except Exception as e:
        print(f"Error: {e}")

# Ejemplo de uso
carpeta = "RealTimeObjectDetection/Tensorflow/workspace/images/collectedimages/Te amo"  # Cambia esto por tu carpeta
base_nombre = "Te amo"  # Cambia esto al nombre base deseado
renombrar_archivos_incrementales(carpeta, base_nombre)
