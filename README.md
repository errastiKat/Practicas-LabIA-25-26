<div align="center">
  <img src="./docs/images/banner.png" alt="Prácticas LabIA 25/26" width="100%"/>
</div>

---

Este repositorio recoge un conjunto de **prácticas aplicadas de Inteligencia Artificial** desarrolladas durante el curso 2025/2026 en la **Universidad de Deusto**.

Las prácticas cubren distintos paradigmas de la IA — desde modelos clásicos hasta enfoques de Deep Learning — combinando fundamentos teóricos, implementación práctica y análisis experimental.  
Cada práctica incluye un **notebook ejecutable** y documentación técnica detallada.


---

## Índice
- [Resumen de prácticas](#resumen-de-prácticas)
- [Descripción de las prácticas](#descripción-de-las-prácticas)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Ejecución de los notebooks](#ejecución-de-los-notebooks)
- [Autora](#autora)

---

## Resumen de prácticas

| Práctica | Tema principal | Enfoque | Enlaces |
|---|---|---|---|
| **P01 — Lógica Difusa** | Segmentación de clientes | Sistemas de inferencia difusa | [README](./P01-%20logica_difusa/README.md) · [Colab](https://colab.research.google.com/drive/1ruykcEmRXXTY-H-4MhMYp8vauJ88DJfW) |
| **P02 — Algoritmos Genéticos** | Optimización combinatoria | Algoritmos evolutivos | [README](./P02-algoritmos_geneticos/README.md) · [Colab](https://colab.research.google.com/drive/1cIxw0WI3K3NtY1zSIRw_XrghZezxTzq5) |
| **P03 — Redes Neuronales Convolucionales** | Visión artificial | CNNs y Transfer Learning | [README](./P03-Image_processing/README.md) · [Colab](https://colab.research.google.com/drive/1jsZr0jfd-R1UQV-Abvkv_dDz1Wik-ysg?usp=sharing) |
| **P04 — Procesamiento de Texto** | NLP | ML clásico y Deep Learning | [README](./P04-Procesamiento_texto/README.md) · [Colab](https://colab.research.google.com/drive/1kGrrRnJA9k_1zleUI0RNllDBPvq1yRqq?usp=sharing) |
| **P05 — Series Temporales** | Análisis financiero | Clustering y modelos latentes | [README](./P05-Series_temporales/README.md) · [Colab](https://drive.google.com/file/d/1EEco7v2E7LW_2GysjM_27587RZR2JE88/view?usp=sharing) |
| **P06 — Aprendizaje por Refuerzo** | Control y decisión | Q-Learning y Deep RL | [README](./P06-Reinforcement_learning/README.md) · [Colab](https://colab.research.google.com/drive/1nY6Swf_hZ8A9Y3v_FOUotUELByYzvGlz?usp=sharing) |

---

## Descripción de las prácticas

### P01 — Lógica Difusa
Implementación de un **sistema de inferencia difusa** aplicado a la segmentación de clientes en un contexto de retail.  
Se definen variables lingüísticas, funciones de pertenencia y reglas difusas para clasificar clientes según su comportamiento de compra.  
El resultado final se exporta en formato estructurado para su posterior análisis.

---

### P02 — Algoritmos Genéticos
Diseño de un **algoritmo genético** para resolver un problema de optimización combinatoria.  
El objetivo es seleccionar subconjuntos óptimos de productos maximizando simultáneamente la rentabilidad y la diversidad.  
Se analizan operadores de selección, cruce, mutación y función de fitness.

---

### P03 — Redes Neuronales Convolucionales
Estudio experimental sobre **clasificación de imágenes** utilizando el conjunto Fashion-MNIST.  
Se comparan modelos entrenados desde cero con enfoques de **Transfer Learning**, incorporando técnicas de data augmentation y análisis de activaciones internas.

---

### P04 — Procesamiento de Texto
Pipeline completo de **procesamiento de lenguaje natural** para la clasificación de textos generados por humanos frente a textos generados por IA.  
Incluye análisis exploratorio, modelos clásicos de Machine Learning, redes LSTM y modelos Transformer, junto con técnicas de interpretación y visualización de atenciones.

---

### P05 — Series Temporales
Análisis de **series temporales financieras** mediante segmentación en subseries y clustering.  
Se emplean ventanas deslizantes, representaciones simbólicas y modelos en espacio latente (VAE) para identificar patrones temporales y perfiles de comportamiento.

---

### P06 — Aprendizaje por Refuerzo
Resolución del entorno **MountainCar-v0** comparando enfoques de **Q-Learning tabular** y **Deep Q-Networks (DQN)**.  
Se estudia el impacto de la discretización del espacio de estados y se analizan métricas de convergencia y rendimiento del agente.

---

## Estructura del repositorio

Cada práctica sigue una organización homogénea:

- `README.md` — explicación teórica, metodología y resultados
- `notebook/` — notebook principal ejecutable
- `docs/` — figuras, imágenes y recursos gráficos
- `data/` — datos de entrada y salida (cuando aplica)

---

## Ejecución de los notebooks

**Google Colab (recomendado)**  
Acceder al enlace correspondiente y ejecutar el notebook directamente en la nube.

**Ejecución local**  
Abrir el notebook desde el directorio `notebook/` utilizando Jupyter Notebook o Visual Studio Code.

---

## Autora
**Katrin Muñoz Errasti**  
Laboratorio de Inteligencia Artificial · Universidad de Deusto
