# SmarTest – AI Exam Trainer

Aplicatie Python pentru generarea si corectarea intrebarilor de examen la cursul de Inteligenta Artificiala.

## Caracteristici

- Generare de intrebari pe patru tipuri principale:
  - Strategia potrivita pentru o problema (N-Queens, Hanoi generalizat, Graph Coloring, Knight's Tour)
  - Jocuri in forma normala – verificare echilibru Nash pur
  - CSP cu Backtracking + MRV + Forward Checking / AC-3
  - Arbore MinMax cu Alpha-Beta – valoarea radacinii si numarul de frunze vizitate
- Interfata grafica moderna (PySide6 / Qt6)
- Generare de PDF pentru:
  - cate un PDF per intrebare
  - test combinat din mai multe intrebari
  - feedback de corectare pentru fiecare raspuns
- Corectare automata pe baza de:
  - lematizare foarte simpla (normalizare text)
  - similaritate intre cuvinte cheie
  - comparare structurata pentru raspunsuri numerice

## Rulare rapida

1. Creeaza si activeaza un virtualenv (optional dar recomandat):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # pe Linux/Mac
   .venv\Scripts\activate       # pe Windows
   ```

2. Instaleaza dependintele:

   ```bash
   pip install -r requirements.txt
   ```

3. Ruleaza aplicatia:

   ```bash
   python main.py
   ```

Daca nu ai PySide6 instalat sau apar erori, verifica versiunea de Python (recomandat 3.10+).

## Structura proiect

- `main.py` – punct de intrare, configureaza si lanseaza GUI-ul
- `smartest/app.py` – *facade* pentru logica de domeniu (generare intrebari, corectare)
- `smartest/core` – modele de domeniu, generatoare de intrebari, solvere, evaluatori
- `smartest/services` – servicii transversale (PDF, procesare text)
- `smartest/gui` – interfata grafica PySide6 (pattern MVC light)

Proiectul foloseste mai multe pattern‑uri OOP:
- **Factory Method / Simple Factory** pentru generarea intrebarilor
- **Strategy** pentru solvere si evaluatori de raspuns
- **Facade** pentru expunerea unei interfete simple peste logica interna (`SmarTestApp`)
- **Dependency Injection** pentru a respecta principiul SOLID *Dependency Inversion*
