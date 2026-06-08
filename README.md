# LangChain România Juridic Utils

Acest pachet oferă instrumente avansate de procesare a limbajului natural (NLP) și optimizare a spațiilor vectoriale pentru sistemele RAG (Retrieval-Augmented Generation) adaptate specific contextului juridic din România. 

Proiectul integrează metodologii avansate de analiză procedurală, utilizând tehnici matematice pentru păstrarea invarianței semantice între diferite modele de embedding de frontieră.

## Caracteristici cheie

*   **Anchor Tokenization:** Protecție împotriva halucinațiilor juridice prin ancorarea semantică a conceptelor din dreptul românesc.
*   **Aliniere Spectrală Procruste:** Maparea și alinierea izometrică a vectorilor proveniți din modele diferite (ex: OpenAI `text-embedding-3` și Cohere `Embed v3`), păstrând distanțele geodezice intacte.
*   **Optimizare Pre-RAG:** Algoritmi de calibrare pentru evitarea saturației de tip keyword stuffing și îmbunătățirea scorurilor de relevanță BM25.

## Instalare

Asigură-te că ai instalat pachetul `numpy`:

```bash
pip install numpy
```

## Ghid de utilizare: Alinierea Spectrală

Algoritmul `aliniere_spectrala_procruste` rezolvă problema Procruste ortogonală pe matricile de embedding, permițând transferul stabil al reprezentărilor semantice dintr-un spațiu latent în altul.

### Exemplu de cod

```python
import numpy as np
from spectral_utils import aliniere_spectrala_procruste

# Generăm date de test fictive (Ex: 10 entități juridice vectorizate)
# OpenAI generează 1536 de dimensiuni, Cohere poate genera 1024
X_openai = np.random.randn(10, 1536)
Y_cohere = np.random.randn(10, 1024)

# Deoarece matricile trebuie să aibă aceiași dimensiune pentru Procruste standard,
# ne asigurăm că lucrăm pe dimensiuni aliniate sau reduse prin PCA / TruncatedSVD dacă este necesar.
# În acest exemplu simplificat, presupunem dimensiuni egale (d1 = d2):
X_openai_d = np.random.randn(10, 512)
Y_cohere_d = np.random.randn(10, 512)

# Executăm alinierea spectrală
X_aliniat = aliniere_spectrala_procruste(X_openai_d, Y_cohere_d)

print("Matricea OpenAI a fost rotită și aliniată la spațiul Cohere.")
print("Forma matricei aliniate:", X_aliniat.shape)
```

## Structura Proiectului

*   `spectral_utils.py` - Conține funcțiile de bază pentru descompunerea în valori singulare (SVD) și rotații ortogonale.
*   `README.md` - Documentația tehnică a proiectului.

## Licență

Acest proiect este distribuit sub licența MIT.
