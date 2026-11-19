# Registro de integridad de los artefactos de entrenamiento

Inventario SHA-256 registrado después de los cambios en la aplicación y las pruebas. También se calcularon los hashes del modelo, el archivo de nombres de clases y el notebook antes y después de cargar el modelo e inspeccionar el notebook; los valores anteriores coincidieron con los que aparecen abajo. El notebook se analizó como JSON en modo de solo lectura; no se ejecutó ninguna celda. Los cambios de implementación y pruebas afectaron únicamente rutas de la aplicación, las pruebas, las dependencias, la documentación y OpenSpec. Ninguna operación de escritura afectó estos artefactos de entrenamiento o evaluación.

| Artefacto | SHA-256 |
|---|---|
| `models/fruitvision_ai_model.keras` | `3dcaa1d4e95ae2d5eef6fd17180bbd4a1c92506e591651ad19a0ed5d90e8e8f1` |
| `models/class_names.json` | `91974a4a50c4f0c4e86ccbc6b5aa2444ab33adbc2b30786374bcb7d2dc75414d` |
| `notebooks/fruitvision_ai_entrenamiento.ipynb` | `08a5bb9d2f8491fd5c9395bb3aea5a2f3aaff2c46af4c721e1b2d47f0466ed12` |
| `data/processed/train.csv` | `b368795514ae1c18f7233ea5dfb735b005d8b1375b6caf1c38cbf488ab90cd0e` |
| `data/processed/validation.csv` | `243a2f30a6dc57f471c5903903b8ba657b48d0719c7c69584ce84add18249005` |
| `data/processed/test.csv` | `a347b2963a75c17dde07af1bd8ec8d076db7610393c389b0d648a18d30d09df8` |
| `data/processed/manifest_clean.csv` | `6308816973ef9cf67e3a93624fcebc4050db9d7273e996a7fc5400604797bd81` |
| `models/training/fruitvision_ai_model.keras` | `3dcaa1d4e95ae2d5eef6fd17180bbd4a1c92506e591651ad19a0ed5d90e8e8f1` |
| `models/training/mejor_modelo_inicial.keras` | `20c7c6c00d10b0b3986cdf54d6833fd935afe521f6732159b6b1474a285afab4` |
| `models/training/mejor_modelo_finetuning.keras` | `252b15e3b783007bd22c0dacde46b8d334c0e86fa2db521995e2a13703e98c9c` |
| `reports/classification_report.csv` | `e155db4f4980f9fcdadae5f1ac56f249174e1d843fe6a22d0a8eb89c809bae61` |
| `reports/confusion_matrix.csv` | `518ddb66156dc79000e60e8b09b74eb56320a23dfd9f6615b90cb51a1f982c1b` |
| `reports/errores_confusion.csv` | `a678698276bda2fe03cb48813b60b9f47686abe014c233d2af570584210eff21` |
| `reports/conflictos_excluidos.csv` | `7fce367f21008a8afe3db54270423cc38e61844b765688b1a991d5e2ff0a5d66` |
| `reports/training_history.csv` | `9572a7f06ae6ea6aa5ccf95756706e742d8194fb2d9e61025ee0d35b2280cf07` |
| `reports/curvas_entrenamiento.png` | `867936f4023b6c40967b99842dbe600a7144bb537f9577d97fdd0da7d9304763` |
| `reports/metrics.json` | `587c449d1ec8e0b93c5d6816bcf0cebb13b7e751c7ab58f0608c2671a1d4787a` |
| `reports/training_metadata/class_names.json` | `91974a4a50c4f0c4e86ccbc6b5aa2444ab33adbc2b30786374bcb7d2dc75414d` |
