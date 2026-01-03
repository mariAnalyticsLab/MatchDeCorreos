# MatchDeCorreos
## ✔️ Descripción del problema
En muchos conjuntos de datos académicos o institucionales, los correos electrónicos de investigadores no siguen un formato uniforme ni coinciden exactamente con sus nombres reales.
Esto dificulta la vinculación automática entre nombres de investigadores y correos electrónicos, especialmente cuando existen abreviaciones, números, errores tipográficos o variaciones en el uso de acentos.
Este proyecto aborda el problema de identificar el correo electrónico más probable asociado a cada investigador, a partir de listas independientes de nombres y correos, utilizando técnicas de comparación difusa de texto (fuzzy matching).

## ✔️ Funcionamiento
El script recibe:
- Un archivo CSV con correos electrónicos no identificados.
- Un archivo CSV con nombres de investigadores.
Se normalizan los textos:
- Conversión a minúsculas.
- Eliminación de acentos.
- Eliminación de caracteres especiales.
- Eliminación de números en los correos.
- Uso únicamente del texto previo al símbolo @.
- Se calcula la similitud entre cada nombre y cada correo utilizando la librería RapidFuzz.
- Se compara cada nombre contra todos los correos en un solo bloque para mejorar el rendimiento.
- Para cada nombre se selecciona el correo con mayor similitud.
- Se aplica un umbral de similitud para clasificar los resultados:
- Coincidencias válidas.
- Casos no encontrados o de baja similitud.
- Los resultados se guardan en archivos CSV separados.

## ✔️ Herramientas y tecnologías utilizadas
- Python 3
- Jupyter Notebook / Script Python
- Librerías
- rapidfuzz
- pandas
- numpy
- unidecode
- re
- time
- os
- csv

## ✔️ Datos
- Este repositorio no incluye datos reales de investigadores.
- Los archivos utilizados como entrada y salida durante la ejecución pueden contener información potencialmente sensible (como correos electrónicos), por lo que no se publican y se usan únicamente en entornos controlados con fines académicos o administrativos.

## ✔️ Resultados y aprendizajes obtenidos
- Emparejamiento automático entre nombres y correos electrónicos con alta precisión.
- Reducción significativa del trabajo manual en procesos de vinculación de datos.
- Uso eficiente de técnicas de fuzzy matching para resolver ambigüedades textuales.
- Implementación de cálculo vectorizado para mejorar el rendimiento.
- Separación clara entre coincidencias confiables y casos dudosos.
- Medición del tiempo de ejecución para evaluar eficiencia.

## ✔️ Limitaciones
- La similitud textual no garantiza una correspondencia real en todos los casos.
- Correos demasiado genéricos pueden generar falsos positivos.
- El resultado depende fuertemente del umbral de similitud definido.
- No se consideran múltiples correos posibles por nombre (solo el mejor candidato).

## ✔️ Disclaimer
- Este proyecto utiliza técnicas de comparación de texto aplicadas a datos previamente disponibles para el usuario, con fines educativos, analíticos y de investigación.
- El uso y tratamiento de la información es responsabilidad del usuario final, quien debe respetar las políticas de privacidad y protección de datos correspondientes.
