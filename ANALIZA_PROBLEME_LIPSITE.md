# Analiză: Probleme care nu pot fi generate de SmarTest

## Rezumat Executiv

Proiectul **SmarTest** poate genera în prezent doar **4 tipuri de probleme**:
1. **STRATEGY_SELECTION** - selectarea strategiei potrivite (N-Queens, Hanoi, Graph Coloring, Knight's Tour)
2. **NASH_EQUILIBRIUM** - verificarea echilibrului Nash pur în jocuri 2x2
3. **CSP_COMPLETION** - completarea CSP cu Backtracking + MRV + Forward Checking
4. **MINIMAX_ALPHA_BETA** - calcularea valorii radăcinii și numărului de frunze evaluate în arbori MinMax cu Alpha-Beta

## Probleme care NU pot fi generate (din lista furnizată)

### 1. Probleme de Căutare (Stări și Euristici)

#### A. Problema cu Matricea 3x3 goală (Plasare X și O)
**Status: ❌ NU este suportată**

**Întrebări care nu pot fi generate:**
- Propuneți o reprezentare pentru o stare a problemei. Justificați alegerea.
- Dacă g(s) și h(s) sunt euristici admisibile, este și $g(s) \cdot 0.5 + h(s) \cdot 0.5$ întotdeauna o euristică admisibilă? Justificați.
- Implementați o tranziție și validarea sa.
- Care este dimensiunea minimă a spațiului problemei?
- Care din strategiile discutate în curs ar fi potrivită pentru această problemă? Justificați.
- Implementați o euristică ce permite strategiei Hillclimbing să găsească soluție.

**Ce lipsește:**
- Generator pentru problema X și O (Tic-Tac-Toe)
- Model de stare pentru matrice 3x3
- Generator de întrebări despre reprezentarea stării
- Generator de întrebări despre euristici admisibile
- Generator de întrebări despre dimensiunea spațiului problemei
- Generator de întrebări despre strategii pentru probleme specifice
- Generator de întrebări despre euristici pentru Hillclimbing

#### B. Problema 8-Puzzle (Matrice 3x3 cu cifre)
**Status: ❌ NU este suportată**

**Întrebări care nu pot fi generate:**
- Propuneți o reprezentare pentru o stare a problemei. Justificați alegerea.
- Cum putem verifica dacă o instanță a problemei permite recuperarea unei soluții?
- Implementați strategia Backtracking pentru recuperarea unei soluții.
- Ce determină dimensiunea spațiului problemei? Propuneți o reprezentare care să minimizeze dimensiunea acestui spațiu.
- Dacă $s(i)$ este valoarea căsuței $i$, este $f(s) = \sum_{i=1}^9 |s(i)-i|$ o euristică admisibilă? Justificați.
- Dați un exemplu de instanță pentru care DFS ar recupera mai repede soluția decât BFS. Justificați.

**Ce lipsește:**
- Generator pentru problema 8-Puzzle
- Model de stare pentru puzzle-uri
- Generator de întrebări despre verificarea solvabilității
- Generator de întrebări despre implementarea Backtracking pentru puzzle-uri
- Generator de întrebări despre analiza dimensiunii spațiului
- Generator de întrebări despre euristici admisibile pentru puzzle-uri
- Generator de întrebări comparative DFS vs BFS

#### C. Problema Camionului (Truck Loading)
**Status: ❌ NU este suportată**

**Întrebări care nu pot fi generate:**
- Propuneți o reprezentare pentru o stare a problemei care să minimizeze dimensiunea spațiului problemei. Câte stări conține acest spațiu?
- Considerând soluție optimă soluția care parcurge cele mai puține noduri, implementați o funcție care verifică dacă o soluție este soluția optimă.
- Propuneți o euristică admisibilă pentru problema dată. Justificați faptul că este admisibilă.
- Propuneți o reprezentare pentru o stare a problemei. Justificați alegerea.
- Cum putem verifica dacă o instanță a problemei permite recuperarea unei soluții?

**Ce lipsește:**
- Generator pentru problema Truck Loading
- Model de stare pentru probleme de încărcare
- Generator de întrebări despre optimizarea reprezentării stării
- Generator de întrebări despre verificarea optimalității soluției
- Generator de întrebări despre euristici admisibile pentru probleme de optimizare

---

### 2. Teoria Jocurilor (MIN-MAX, Nash)

#### Strategia MIN-MAX
**Status: ⚠️ PARȚIAL suportată**

**Ce este suportat:**
- ✅ Arbore MinMax cu Alpha-Beta – valoarea radăcinii și numărul de frunze vizitate

**Ce NU este suportat:**
- ❌ Dacă problema cu Matricea 3x3 goală este reformulată ca joc, propuneți o euristică ce permite strategiei MIN-MAX să funcționeze ca mecanism de decizie pentru primul jucător.
- ❌ Dați exemplu de arbore MINIMAX pe două nivele (MIN și MAX) în care optimizarea AlphaBeta ar elimina exact un test suplimentar.

**Ce lipsește:**
- Generator de întrebări despre euristici pentru MinMax în jocuri specifice
- Generator de întrebări despre analiza optimizării Alpha-Beta (câte noduri sunt eliminate)

#### Strategii Dominante și Echilibre Nash
**Status: ⚠️ PARȚIAL suportată**

**Ce este suportat:**
- ✅ Verificare echilibru Nash pur (doar pentru jocuri 2x2)

**Ce NU este suportată:**
- ❌ Pentru un joc dat (Daffy vs Bugs), există strategii pure dominante pentru unul din jucători? Dar strategii strict dominante?
- ❌ Pentru un joc dat (Mario vs Luigi), există strategii dominate pentru cel puțin unul din jucători? Dar echilibre Nash pure?
- ❌ Pentru un joc dat (A/B vs a/b/c), există echilibre Nash pure? Justificați.

**Ce lipsește:**
- Generator de întrebări despre strategii dominante
- Generator de întrebări despre strategii strict dominante
- Generator de întrebări despre strategii dominate
- Suport pentru jocuri cu mai mult de 2 strategii per jucător
- Generator de întrebări despre jocuri cu nume specifice (Daffy vs Bugs, Mario vs Luigi)

#### Jocuri (Definiții)
**Status: ❌ NU este suportată**

**Întrebări care nu pot fi generate:**
- Sugerați modificări pentru enunțul problemei Camionului astfel încât ea să devină un joc (problemă interactivă de decizie).

**Ce lipsește:**
- Generator de întrebări despre transformarea problemelor în jocuri
- Generator de întrebări despre definirea jocurilor

---

### 3. Probleme de Satisfacere a Restricțiilor (CSP)

**Status: ⚠️ PARȚIAL suportată**

**Ce este suportat:**
- ✅ CSP cu Backtracking + MRV + Forward Checking (completarea asignărilor)

**Ce NU este suportată:**

#### Consistența Arc (Arc Consistency)
- ❌ Aplicați algoritmul de consistență arc pentru a actualiza domeniile variabilelor X, Y, Z (date: $X > Y, Y + Z = 12, X + Z = 16$).
- ❌ Aplicați algoritmul Arc consistency pentru a actualiza valorile variabilelor v1, v2, v3 (graf de restricții). Specificați dacă problema este inconsistentă.

**Ce lipsește:**
- Generator de întrebări despre algoritmul AC-3 (Arc Consistency)
- Generator de întrebări despre actualizarea domeniilor prin arc consistency
- Generator de întrebări despre verificarea inconsistenței prin AC-3
- Suport pentru constrângeri numerice ($X > Y$, $Y + Z = 12$, etc.)

#### Forward Checking (FC) și MRV
- ✅ Parțial: există FC + MRV pentru completarea CSP-urilor
- ❌ Aplicați algoritmul Forward checking (FC) pentru problema celor 4-regine, având asignarea parțială $Q1=2$. Identificați o soluție sau inconsistența.
- ❌ Utilizați euristica MRV și algoritmul FC pentru a colora o hartă cu 4 regiuni. Precizați care este complexitatea algoritmului de căutare pentru o hartă cu $d$ regiuni și $m$ culori posibile.

**Ce lipsește:**
- Generator de întrebări despre aplicarea pas cu pas a FC pentru probleme specifice (4-regine)
- Generator de întrebări despre analiza complexității algoritmilor CSP
- Generator de întrebări despre probleme de colorare a hărților

---

### 4. Procese de Decizie Markov (MDP) și Învățare prin Întărire (RL)

**Status: ❌ COMPLET NESUPORTAT**

#### Value Iteration și Policy Iteration
**Întrebări care nu pot fi generate:**
- Actualizați valorile utilităților utilizând algoritmul de iterare a valorilor (Value iteration) pentru o configurație de grid dată.
- Care stări își vor actualiza valoarea funcției utilitate în urma utilizării unui pas al algoritmului Value iteration? (Grid dat)
- Calculați valoarea utilității unei stări (stânga-jos) considerând încă un pas al algoritmului Value iteration. Care este politica recomandată în această stare?
- Care este complexitatea unei iterații a algoritmului Value iteration?
- Prin ce diferă algoritmii Policy iteration și Value iteration? Care converge mai rapid?

**Ce lipsește:**
- Generator pentru probleme MDP
- Model pentru grid-uri MDP
- Generator de întrebări despre Value Iteration
- Generator de întrebări despre Policy Iteration
- Generator de întrebări despre calcularea utilităților
- Generator de întrebări despre determinarea politicii optime
- Generator de întrebări despre analiza complexității algoritmilor MDP
- Generator de întrebări comparative între algoritmi

#### Q-learning și TD-learning
**Întrebări care nu pot fi generate:**
- Actualizați valorile Q utilizând algoritmul Q-learning pentru secvențe de tranziții date.
- Actualizați valorile utilităților utilizând algoritmul de învățare a diferențelor temporale (TD-learning) pentru secvențe de tranziții date.
- Calculați valorile Q pentru problema Pacman, considerând două încercări (observații date).
- Actualizați valorile Q pentru o serie de observații de tip $(s, a, s', R(s,a,s'))$. Care este politica recomandată pentru o anumită stare?
- Care este rolul parametrilor $\alpha$, $\gamma$ și $\epsilon$ din cadrul metodei $\epsilon$-greedy Q-learning? Care este efectul individual al setării fiecăruia la valoarea 0?

**Ce lipsește:**
- Generator pentru probleme de Reinforcement Learning
- Model pentru secvențe de tranziții $(s, a, s', R)$
- Generator de întrebări despre Q-learning
- Generator de întrebări despre TD-learning
- Generator de întrebări despre probleme specifice (Pacman)
- Generator de întrebări despre parametrii algoritmilor RL ($\alpha$, $\gamma$, $\epsilon$)

---

### 5. Rețele Bayesiane

**Status: ❌ COMPLET NESUPORTAT**

**Întrebări care nu pot fi generate:**
- Calculați probabilitatea marginală P(C) (rețea bayesiană dată).
- Calculați probabilitatea de a ploua dacă cerul este înnorat și iarba nu este udă, folosind metoda de inferență prin enumerare (rețea bayesiană dată).
- Dacă pe gazon sunt păsări, dar nu sunt râme afară, care este probabilitatea să fie primăvară? (Rețea și tabele de probabilități condiționale date).
- Calculați următoarele probabilități marginale: a) de a ploua b) de a fi râme afară c) de a fi păsări pe gazon (Rețea și tabele de probabilități condiționale date).
- Precizați cel puțin două relații de independență condițională din rețeaua bayesiană. Justificați răspunsul.

**Ce lipsește:**
- Generator pentru rețele bayesiene
- Model pentru grafuri bayesiene (noduri, muchii, tabele de probabilități condiționale)
- Generator de întrebări despre calcularea probabilităților marginale
- Generator de întrebări despre inferență prin enumerare
- Generator de întrebări despre probabilități condiționate
- Generator de întrebări despre independență condițională
- Generator de întrebări despre analiza structurii rețelei

---

### 6. Probleme de Planificare (STRIPS/ADL)

**Status: ❌ COMPLET NESUPORTAT**

#### Descrierea Operațiilor (STRIPS/ADL)
**Întrebări care nu pot fi generate:**
- Descrieți operațiile Go(there) și Buy(x) în limbajul STRIPS/ADL.
- Descrieți operațiile FromTable(x, y) și ToTable(x, y) în limbajul STRIPS/ADL.
- Descrieți operațiile Buy(x, store) și Go(x, y) în limbajul STRIPS/ADL.
- Descrieți operațiile PlaceCap(), RemoveCap(), Insert(i) în limbajul STRIPS/ADL.

**Ce lipsește:**
- Generator pentru probleme de planificare
- Model pentru operații STRIPS/ADL (precondiții, efecte, parametri)
- Generator de întrebări despre descrierea operațiilor în STRIPS/ADL
- Generator de întrebări despre predicate și acțiuni

#### Planificare cu Ordine Parțială (POP)
**Întrebări care nu pot fi generate:**
- Construiți un plan incomplet (minim 3 acțiuni) utilizând algoritmul de planificare cu ordine parțială pentru obiective date.
- Descrieți stările inițiale și obiectivele utilizând predicate.

**Ce lipsește:**
- Generator pentru probleme POP (Partial Order Planning)
- Model pentru planuri parțiale (acțiuni, ordine, legături)
- Generator de întrebări despre construirea planurilor parțiale
- Generator de întrebări despre reprezentarea stărilor și obiectivelor cu predicate

---

### 7. Ontologii și Lingvistică Computațională

**Status: ❌ COMPLET NESUPORTAT**

#### Ontologii
**Întrebări care nu pot fi generate:**
- Există situații în care o parte din cunoașterea specifică unui domeniu nu poate fi reprezentată într-o ontologie? Justificați și exemplificați.
- Care este rolul instanțelor într-o ontologie? Exemplificați.
- Care este rolul inferențelor într-o ontologie? Exemplificați.
- Descrieți o relație semantică cu cel puțin trei membrii ce ar putea fi inclusă într-o ontologie pentru Facultatea de Informatică. Descrieți și o posibilă inferență.

**Ce lipsește:**
- Generator pentru probleme despre ontologii
- Model pentru ontologii (clase, instanțe, proprietăți, relații)
- Generator de întrebări despre limitările ontologiilor
- Generator de întrebări despre rolul instanțelor
- Generator de întrebări despre rolul inferențelor
- Generator de întrebări despre relații semantice

#### Lingvistică Computațională
**Întrebări care nu pot fi generate:**
- Cum putem identifica autorul unui text ca fiind una din două persoane dacă avem la dispoziție exemple numeroase? Indicați și tehnologiile necesare specifice Lingvisticii Computaționale.

**Ce lipsește:**
- Generator pentru probleme de lingvistică computațională
- Generator de întrebări despre identificarea autorului
- Generator de întrebări despre tehnologii NLP (Natural Language Processing)
- Generator de întrebări despre analiza textului

---

## Rezumat Numeric

### Probleme suportate: 4 tipuri
1. Strategy Selection (4 probleme: N-Queens, Hanoi, Graph Coloring, Knight's Tour)
2. Nash Equilibrium (doar echilibru Nash pur, jocuri 2x2)
3. CSP Completion (Backtracking + MRV + Forward Checking)
4. Minimax Alpha-Beta (arbori de joc)

### Probleme NESUPORTATE: ~80+ tipuri de întrebări

**Categorii complet nesuportate:**
- ❌ Probleme de căutare specifice (X și O, 8-Puzzle, Truck Loading)
- ❌ MDP și RL (Value Iteration, Policy Iteration, Q-learning, TD-learning)
- ❌ Rețele Bayesiane
- ❌ Planificare (STRIPS/ADL, POP)
- ❌ Ontologii și Lingvistică Computațională

**Categorii parțial suportate:**
- ⚠️ Teoria Jocurilor (doar Nash pur 2x2, lipsesc strategii dominante, jocuri mai complexe)
- ⚠️ CSP (doar FC+MRV pentru completare, lipsesc AC-3, analiza complexității, probleme specifice)

---

## Recomandări pentru Extindere

Pentru a acoperi toate problemele din listă, ar trebui adăugate:

1. **Generatori noi pentru:**
   - Probleme de căutare specifice (X și O, 8-Puzzle, Truck Loading)
   - MDP și RL
   - Rețele Bayesiane
   - Planificare STRIPS/ADL
   - Ontologii și NLP

2. **Extinderi pentru generatoare existente:**
   - Nash: strategii dominante, jocuri mai complexe
   - CSP: AC-3, analiza complexității, probleme specifice (4-regine, hărți)
   - MinMax: euristici pentru jocuri specifice, analiza optimizării

3. **Modele noi de date:**
   - Stări pentru puzzle-uri și jocuri
   - Grid-uri MDP
   - Rețele bayesiene
   - Planuri și operații STRIPS
   - Ontologii

4. **Solvere noi:**
   - Solvere pentru MDP (Value Iteration, Policy Iteration)
   - Solvere pentru RL (Q-learning, TD-learning)
   - Solvere pentru rețele bayesiene (inferență)
   - Solvere pentru planificare (POP)

5. **Evaluatori noi:**
   - Evaluatori pentru răspunsuri despre probabilități
   - Evaluatori pentru răspunsuri despre planuri
   - Evaluatori pentru răspunsuri despre ontologii

