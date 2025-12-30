# GUI Update - MDP și RL Support

## Noi Opțiuni în Interfața Grafică

Am adăugat **5 checkbox-uri noi** în tab-ul "Question Generator" pentru a genera întrebări de tip MDP și Reinforcement Learning:

### Checkbox-uri Noi (în ordinea apariției):

#### Probleme Clasice (existente):
1. ✅ **Strategie pentru problema (N-Queens, etc.)**
2. ✅ **Echilibru Nash (joc 2x2)**
3. ✅ **CSP cu Backtracking + MRV**
4. ✅ **MinMax cu Alpha-Beta**

#### Probleme MDP și RL (NOI):
5. ✅ **Value Iteration (MDP)** - Probleme de iterare a valorilor pe grid worlds MDP
6. ✅ **Policy Iteration (MDP)** - Întrebări comparative și probleme de iterare a politicii
7. ✅ **Q-learning (RL)** - Probleme de Q-learning cu secvențe de tranziții
8. ✅ **TD-learning (RL)** - Probleme de TD(0) learning
9. ✅ **Parametri RL (alpha, gamma, epsilon)** - Întrebări despre rolul parametrilor RL

## Cum să Folosești Noile Funcționalități

### 1. Generare Întrebări

1. Pornește aplicația: `python main.py`
2. În tab-ul "Question Generator", selectează tipurile de întrebări dorite
3. Setează numărul de întrebări (1-20)
4. Click pe "Genereaza intrebari"

**Exemplu**: Selectează "Value Iteration (MDP)" și "Q-learning (RL)" pentru a genera întrebări mixte MDP/RL.

### 2. Rezolvare Probleme în Limbaj Natural

1. Mergi la tab-ul "Rezolvator Probleme Q&A"
2. Scrie problema în română sau engleză
3. Click pe "Rezolva problema"

**Exemple de probleme suportate**:

```
Grid 3x4 cu gamma=0.9, reward la (0,3)=1.0 si (1,3)=-1.0, 
perete la (1,1). Aplica un pas de value iteration.
```

```
Q-learning cu alpha=0.1, gamma=0.9. 
Tranzitie: s=(0,0), a=right, s'=(0,1), r=0.5
```

```
MDP grid world 3x4 with value iteration
```

### 3. Evaluare Răspunsuri

1. Selectează o întrebare din listă
2. Scrie răspunsul tău în câmpul "Raspunsul tau"
3. Click pe "Evalueaza raspunsul"
4. Primești score și feedback automat

## Screenshot-uri Conceptuale

### Panoul de Generare (stânga):
```
┌─────────────────────────────────────┐
│  Generare intrebari                 │
│                                     │
│  Numar de intrebari: [4]           │
│                                     │
│  ☑ Strategie pentru problema       │
│  ☑ Echilibru Nash (joc 2x2)       │
│  ☑ CSP cu Backtracking + MRV      │
│  ☑ MinMax cu Alpha-Beta           │
│  ☑ Value Iteration (MDP)          │  ← NOU
│  ☑ Policy Iteration (MDP)         │  ← NOU
│  ☑ Q-learning (RL)                │  ← NOU
│  ☑ TD-learning (RL)               │  ← NOU
│  ☑ Parametri RL (alpha, gamma)   │  ← NOU
│                                     │
│  [Genereaza intrebari]             │
└─────────────────────────────────────┘
```

### Lista Întrebări Generate:
```
┌─────────────────────────────────────┐
│ Intrebari generate:                 │
│                                     │
│ • [VALUE_ITERATION] Value Iteration│
│   pe Grid World                     │
│ • [Q_LEARNING] Q-learning cu       │
│   secventa de tranzitii             │
│ • [NASH_EQUILIBRIUM] Echilibru     │
│   Nash pur intr-un joc 2x2         │
│ • [CSP_COMPLETION] CSP cu          │
│   Backtracking + MRV + FC          │
└─────────────────────────────────────┘
```

## Tipuri de Întrebări Generate

### Value Iteration (MDP)
- Grid world 3x4 sau 4x3
- Recompense: pozitive (goal), negative (penal), living cost
- Parametri: γ (discount factor), probabilități de tranziție
- Întrebări: calculează V(s), extrage politica, identifică stări actualizate

### Policy Iteration (MDP)
- Comparații cu Value Iteration
- Întrebări despre convergență
- Analiza complexității

### Q-learning (RL)
- Secvențe de tranziții: (s, a, s', r)
- Parametri: α=0.1, γ=0.9, ε=0.1
- Întrebări: calculează Q-values, extrage politica

### TD-learning (RL)
- Secvențe de observații: (s, s', r)
- Formula TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]
- Întrebări: calculează V(s), TD-errors

### Parametri RL
- Întrebări conceptuale despre α, γ, ε
- Efectele setării la 0
- Trade-off-uri (explorare vs exploatare)

## Export și Import

### Export Opțiuni:
1. **Export test complet** - Toate întrebările într-un singur PDF
2. **Export evaluare** - Feedback-ul pentru un răspuns specific

### Import:
- **Import răspuns din PDF** - Încarcă răspunsul din fișier PDF pentru evaluare

## Beneficii

✅ **Generare Rapidă**: Generează 4-20 întrebări în < 1 secundă
✅ **Varietate**: 10 tipuri diferite de probleme AI
✅ **Evaluare Automată**: Score și feedback instant
✅ **Q&A Natural**: Rezolvă probleme în limbaj natural
✅ **Export PDF**: Creează materiale de studiu

## Note Tehnice

- Toate checkbox-urile sunt bifate implicit (toate tipurile de întrebări vor fi generate)
- Poți deselecta tipurile care nu te interesează
- Numărul de întrebări este distribuit uniform între tipurile selectate
- Fiecare întrebare are dificultate "medium" by default
- Răspunsurile sunt evaluate cu toleranță numerică pentru valori flotante (±5%)

## Support

Pentru probleme sau sugestii:
- Verifică console pentru erori
- Rulează testele: `python3 tests/test_mdp_rl.py`
- Citește documentația completă în `MDP_RL_IMPLEMENTATION_SUMMARY.md`

