# SmarTest – AI Exam Trainer

## Prezentare Generală

SmarTest este o aplicație Python comprehensivă pentru generarea, rezolvarea și evaluarea automată a întrebărilor de examen la cursul de Inteligență Artificială. Aplicația acoperă un spectru larg de subiecte din AI clasică, incluzând algoritmi de căutare, teoria jocurilor, satisfacerea constrângerilor, procese decizionale Markov, învățare prin întărire și planificare clasică.

### Capabilități Principale

- **13 tipuri de întrebări** acoperind toate domeniile majore ale AI
- **Generare automată** cu randomizare pentru variabilitate
- **Rezolvare automată** folosind algoritmi specifici fiecărui domeniu
- **Evaluare inteligentă** cu scoring semantic și feedback detaliat
- **Interfață grafică modernă** construită cu PySide6 (Qt6)
- **Export PDF** pentru întrebări, teste și feedback
- **Sistem Q&A** pentru răspunsuri la întrebări în limbaj natural

---

## Arhitectura Aplicației

### Structura de Fișiere

**Punct de Intrare:**
- `main.py` - Lansează aplicația, inițializează GUI-ul și logica de domeniu

**Core (Logica Principală):**
- `smartest/core/models.py` - Modele de date pentru toate tipurile de probleme
- `smartest/core/generators.py` - Generatori de întrebări pentru fiecare tip
- `smartest/core/solvers.py` - Algoritmi de rezolvare pentru fiecare tip
- `smartest/core/evaluators.py` - Evaluatori de răspunsuri cu scoring semantic

**Services (Servicii):**
- `smartest/services/qa_service.py` - Orchestrare parsing → solving → formatting
- `smartest/services/problem_parser.py` - Extragere structurată din limbaj natural
- `smartest/services/text_processing.py` - Normalizare text, lematizare simplă
- `smartest/services/pdf_service.py` - Generare documente PDF

**GUI (Interfață Grafică):**
- `smartest/gui/main_window.py` - Fereastră principală cu generare și evaluare
- `smartest/gui/qa_panel.py` - Panel interactiv pentru Q&A în limbaj natural

**Facade:**
- `smartest/app.py` - Interfață simplificată peste toată logica aplicației

### Pattern-uri de Design

**Factory Method:**
- `QuestionFactory` - Creează instanțe de generatori bazat pe tipul întrebării
- `SolverFactory` - Creează instanțe de solvere bazat pe tipul problemei
- `EvaluatorFactory` - Creează instanțe de evaluatori bazat pe tipul întrebării

**Strategy Pattern:**
- Fiecare solver implementează interfața `Solver` cu metoda abstractă `solve()`
- Fiecare evaluator implementează `AnswerEvaluator` cu metoda `evaluate()`
- Permite adăugarea de noi tipuri fără modificarea codului existent

**Facade Pattern:**
- `SmarTestApp` oferă o interfață simplă peste complexitatea internă
- Ascunde detaliile de coordonare între generatori, solvere, evaluatori

**Dependency Injection:**
- Componentele primesc dependențele prin constructor (ex: `TextProcessor`)
- Respectă principiul SOLID de Dependency Inversion

---

## Tipurile de Întrebări Suportate

### 1. STRATEGY_SELECTION - Selectarea Strategiei

**Domeniu:** Algoritmi de căutare și rezolvare de probleme

**Descriere Detaliată:**
Acest tip de întrebare prezintă o problemă clasică de AI (N-Queens, Turnurile din Hanoi, Colorarea Grafurilor sau Knight's Tour) și cere studentului să identifice strategia de căutare cea mai potrivită pentru rezolvarea ei. Întrebarea include o descriere completă a problemei, specificațiile parametrilor (dimensiunea tablei, numărul de discuri, numărul de culori etc.) și opțiunile de strategii disponibile.

**Probleme Incluse:**

**N-Queens:**
- Plasarea a N regine pe o tablă de șah N×N astfel încât nicio regină să nu atace o altă regină
- Dimensiune tablă: randomizată între 4×4 și 8×8
- Variabilitate: diferite dimensiuni de tablă în fiecare generare
- Strategie recomandată: Backtracking cu verificare constrângeri

**Turnurile din Hanoi Generalizat:**
- Mutarea discurilor între tije respectând regula că un disc mai mare nu poate fi peste unul mai mic
- Număr tije: randomizat între 3 și 4
- Număr discuri: randomizat între 3 și 8
- Variabilitate: combinații diferite de tije și discuri
- Strategie recomandată: DFS (Depth-First Search) datorită structurii recursive

**Graph Coloring:**
- Colorarea nodurilor unui graf astfel încât nodurile adiacente să aibă culori diferite
- Număr noduri: randomizat între 4 și 6
- Muchii: generate aleator cu densități diferite
- Număr culori: randomizat între 3 și 4
- Variabilitate: grafuri diferite la fiecare generare
- Strategie recomandată: Backtracking cu MRV (Minimum Remaining Values) și Forward Checking

**Knight's Tour:**
- Găsirea unei rute pentru un cal de șah care vizitează fiecare pătrat exact o dată
- Dimensiune tablă: randomizată între 5×5, 6×6 și 8×8
- Poziție start: randomizată
- Variabilitate: diferite dimensiuni și puncte de start
- Strategie recomandată: Backtracking cu euristici (Warnsdorff's rule)

### 2. NASH_EQUILIBRIUM - Echilibru Nash

**Domeniu:** Teoria Jocurilor

**Descriere Detaliată:**
Acest tip de întrebare prezintă un joc în formă normală (matriceală) cu doi jucători și cere identificarea echilibrelor Nash pure. Un echilibru Nash este o combinație de strategii în care niciun jucător nu poate îmbunătăți unilateral payoff-ul său prin schimbarea strategiei, presupunând că celălalt jucător își păstrează strategia.

**Caracteristici:**
- Jocuri 2×2 (două strategii pentru fiecare jucător)
- Payoff-uri randomizate între 0 și 10
- Matricea de payoff-uri afișată clar: (payoff_jucător_rând, payoff_jucător_coloană)
- Verificare sistematică pentru toate cele 4 combinații posibile
- Pot exista 0, 1 sau 2 echilibre Nash pure

**Algoritmul de Identificare:**
Pentru fiecare combinație de strategii, se verifică dacă:
- Jucătorul rând nu poate obține un payoff mai mare schimbând strategia (celălalt rămâne fix)
- Jucătorul coloană nu poate obține un payoff mai mare schimbând strategia (celălalt rămâne fix)
- Dacă ambele condiții sunt îndeplinite, avem un echilibru Nash

### 3. CSP_COMPLETION - Satisfacerea Constrângerilor

**Domeniu:** Constraint Satisfaction Problems

**Descriere Detaliată:**
Acest tip de întrebare prezintă o problemă de satisfacere a constrângerilor (CSP) și cere completarea asignărilor variabilelor folosind Backtracking cu euristicile MRV (Minimum Remaining Values) și Forward Checking. Problema include variabile, domenii, constrângeri și eventual o asignare parțială inițială.

**Caracteristici:**
- Număr variabile: randomizat între 3 și 5
- Domenii: valori întregi sau culori (roșu, albastru, verde, galben)
- Constrângeri: relații de inegalitate între perechi de variabile
- Asignare parțială: 0-2 variabile pre-asignate
- Variabilitate: diferite configurații de variabile, domenii și constrângeri

**Algoritm Backtracking + MRV + Forward Checking:**

**MRV (Minimum Remaining Values):**
- Selectează următoarea variabilă de asignat ca fiind cea cu cele mai puține valori legale rămase
- Principiu: "fail-first" - detectează inconsistențele mai devreme
- Reduce dramatic spațiul de căutare

**Forward Checking:**
- După fiecare asignare, actualizează domeniile variabilelor neassignate
- Elimină valorile care ar încălca constrângerile cu variabila tocmai asignată
- Detectează early failure când un domeniu devine gol

**Backtracking:**
- Dacă o asignare duce la inconsistență (domeniu gol), revine la pasul anterior
- Încearcă altă valoare pentru variabila curentă
- Continuă până găsește soluție sau epuizează toate posibilitățile

### 4. MINIMAX_ALPHA_BETA - MinMax cu Alpha-Beta Pruning

**Domeniu:** Arbori de joc și strategii adversariale

**Descriere Detaliată:**
Acest tip de întrebare prezintă un arbore de joc cu niveluri MAX și MIN alternante și cere calcularea valorii nodului rădăcină folosind algoritmul Minimax cu optimizarea Alpha-Beta Pruning, precum și numărul de frunze evaluate (nu tăiate).

**Caracteristici:**
- Adâncime arbore: randomizată între 2 și 4 niveluri
- Branching factor: randomizat între 2 și 3 copii per nod
- Valori frunze: randomizate între -10 și 10
- Structura: niveluri MAX și MIN alternante (rădăcina este MAX)
- Variabilitate: arbori diferiți la fiecare generare

**Algoritmul Minimax:**
- MAX încearcă să maximizeze scorul
- MIN încearcă să minimizeze scorul
- Propagare bottom-up: valoarea unui nod este determinată de copiii săi
- Nod MAX: ia maximul valorilor copiilor
- Nod MIN: ia minimul valorilor copiilor

**Alpha-Beta Pruning:**
- Alpha: cea mai bună valoare pentru MAX găsită până acum (lower bound)
- Beta: cea mai bună valoare pentru MIN găsită până acum (upper bound)
- Pruning: dacă Beta ≤ Alpha, ramura curentă poate fi tăiată (nu influențează decizia)
- Optimizare: reduce numărul de noduri evaluate de la O(b^d) la O(b^(d/2)) în cel mai bun caz

**Contorizare Frunze:**
- Se numără doar frunzele efectiv evaluate
- Frunzele din ramuri tăiate (pruned) NU sunt numărate
- Rezultatul arată eficiența Alpha-Beta vs Minimax simplu

### 5. VALUE_ITERATION - Iterarea Valorilor (MDP)

**Domeniu:** Procese Decizionale Markov (Markov Decision Processes)

**Descriere Detaliată:**
Acest tip de întrebare prezintă un grid world (lume pe grilă) cu stări, recompense, stări terminale și un discount factor, cerând calcularea valorilor utilităților pentru fiecare stare folosind algoritmul Value Iteration și determinarea politicii optime.

**Caracteristici Grid World:**
- Dimensiune: randomizată între 2×2 și 4×4
- Stări terminale: 1-2 stări cu recompense mari (+1 sau -1)
- Stări normale: recompensă mică negativă (de obicei -0.04 pentru a încuraja găsirea rapidă a terminalelor)
- Obstacole (ziduri): 0-2 celule blocate aleator
- Discount factor (gamma): randomizat între 0.8 și 0.95
- Probabilități de tranziție: stochastice (ex: 80% direcția dorită, 10% perpendicular stânga, 10% perpendicular dreapta)

**Algoritmul Value Iteration:**

**Ecuația Bellman:**
Pentru fiecare stare s, valoarea utilității V(s) se calculează astfel:
V(s) = max_a [suma peste s' din P(s'|s,a) × (R(s,a,s') + gamma × V(s'))]

Unde:
- a = acțiunea (up, down, left, right)
- s' = starea următoare
- P(s'|s,a) = probabilitatea de a ajunge în s' din s prin acțiunea a
- R(s,a,s') = recompensa imediată
- gamma = discount factor (ponderarea recompenselor viitoare)

**Proces Iterativ:**
1. Inițializare: toate valorile V(s) = 0
2. Iterare până la convergență (schimbarea maximă < epsilon, ex: 0.001):
   - Pentru fiecare stare non-terminală:
   - Calculează noua valoare folosind ecuația Bellman
   - Actualizează V(s)
3. Convergență atinsă când valorile se stabilizează

**Extragerea Politicii:**
După convergență, politica optimă π*(s) pentru fiecare stare este:
π*(s) = argmax_a [suma peste s' din P(s'|s,a) × (R(s,a,s') + gamma × V(s'))]
Adică: alege acțiunea care maximizează valoarea așteptată

**Complexitate:**
O(S² × A) per iterație, unde S = număr stări, A = număr acțiuni
Converge în O(S²A/epsilon) timp total

### 6. POLICY_ITERATION - Iterarea Politicii (MDP)

**Domeniu:** Procese Decizionale Markov

**Descriere Detaliată:**
Similar cu Value Iteration, dar folosește o abordare diferită prin iterarea directă a politicii. Algoritmul alternează între evaluarea politicii curente și îmbunătățirea ei până când politica nu mai poate fi îmbunătățită.

**Diferențe față de Value Iteration:**
- Value Iteration: actualizează valorile direct până la convergență
- Policy Iteration: evaluează complet o politică, apoi o îmbunătățește

**Algoritmul Policy Iteration:**

**Faza 1: Evaluarea Politicii (Policy Evaluation)**
Dată o politică π, calculează V^π(s) pentru toate stările:
V^π(s) = suma peste s' din P(s'|s,π(s)) × (R(s,π(s),s') + gamma × V^π(s'))

Iterează până când valorile converg (diferența maximă < epsilon)

**Faza 2: Îmbunătățirea Politicii (Policy Improvement)**
Pentru fiecare stare s, actualizează politica:
π'(s) = argmax_a [suma peste s' din P(s'|s,a) × (R(s,a,s') + gamma × V^π(s'))]

**Bucla Principală:**
1. Inițializare: politică arbitrară (ex: toate stările aleg "up")
2. Repetă:
   - Evaluare: calculează V^π pentru politica curentă π
   - Îmbunătățire: creează π' bazat pe V^π
   - Dacă π' = π (politica nu s-a schimbat), STOP - am găsit politica optimă
   - Altfel: π ← π', continuă

**Convergență:**
- Policy Iteration converge în mai puține iterații decât Value Iteration
- Fiecare iterație este mai costisitoare (evaluare completă)
- În practică, adesea mai rapid pentru probleme mici/medii

### 7. Q_LEARNING - Învățare Q (Reinforcement Learning)

**Domeniu:** Învățare prin Întărire (Reinforcement Learning)

**Descriere Detaliată:**
Acest tip de întrebare prezintă o secvență de tranziții (observații) într-un mediu și cere actualizarea valorilor Q folosind algoritmul Q-learning. Q-learning este o metodă model-free (fără model explicit al mediului) de învățare a valorilor acțiune-stare.

**Caracteristici:**
- Tranziții: secvențe de (stare, acțiune, stare_următoare, recompensă)
- Număr tranziții: randomizat între 3 și 8
- Stări: coordonate (row, col) într-un grid
- Acțiuni: up, down, left, right
- Recompense: randomizate între -1 și +1
- Parametri RL: alpha (learning rate), gamma (discount), epsilon (exploration)

**Parametri RL:**

**Alpha (Learning Rate) - randomizat între 0.1 și 0.5:**
- Controlează cât de mult învățăm din experiențele noi
- Alpha mic: învățare lentă, stabilitate mare
- Alpha mare: învățare rapidă, volatilitate mare

**Gamma (Discount Factor) - randomizat între 0.8 și 0.95:**
- Ponderează recompensele viitoare
- Gamma aproape de 0: agent miop (valorează doar recompense imediate)
- Gamma aproape de 1: agent cu viziune lungă

**Epsilon (Exploration Rate) - randomizat între 0.1 și 0.3:**
- Balanța exploatare vs explorare
- Cu probabilitate epsilon: acțiune aleatoare (explorare)
- Cu probabilitate 1-epsilon: acțiunea optimă conform Q (exploatare)

**Formula de Actualizare Q-learning:**
Q(s,a) ← Q(s,a) + alpha × [r + gamma × max_a' Q(s',a') - Q(s,a)]

Unde:
- (s,a) = perechea stare-acțiune curentă
- r = recompensa primită
- s' = starea următoare
- max_a' Q(s',a') = valoarea Q maximă în starea următoare (peste toate acțiunile)

**Temporal Difference (TD) Error:**
TD-error = r + gamma × max_a' Q(s',a') - Q(s,a)
- Măsoară discrepanța între estimarea curentă și noua informație
- Dacă TD-error > 0: am subestimat valoarea, creștem Q
- Dacă TD-error < 0: am supraeestimat valoarea, scădem Q

**Off-Policy Learning:**
Q-learning este off-policy: învață politica optimă (greedy) chiar dacă urmează o politică de explorare (epsilon-greedy)

### 8. TD_LEARNING - TD(0) Learning (Reinforcement Learning)

**Domeniu:** Învățare prin Întărire - Diferențe Temporale

**Descriere Detaliată:**
Similar cu Q-learning, dar învață direct valorile stărilor V(s) în loc de valorile acțiune-stare Q(s,a). TD(0) este metoda de bază din familia metodelor Temporal Difference.

**Diferențe față de Q-learning:**
- Q-learning: învață Q(s,a) - valoarea unei acțiuni într-o stare
- TD-learning: învață V(s) - valoarea unei stări indiferent de acțiune
- Q-learning: poate deriva politica direct din Q
- TD-learning: necesită modelul tranziției pentru a extrage politica

**Formula de Actualizare TD(0):**
V(s) ← V(s) + alpha × [r + gamma × V(s') - V(s)]

Unde:
- V(s) = valoarea estimată a stării curente
- r = recompensa primită
- V(s') = valoarea estimată a stării următoare
- alpha, gamma = parametri ca la Q-learning

**TD-error pentru TD(0):**
TD-error = r + gamma × V(s') - V(s)
- Diferența între estimarea bootstrap (r + gamma × V(s')) și estimarea curentă V(s)

**Bootstrapping:**
TD învață din estimări ale altor estimări (folosește V(s') care este el însuși o estimare)
Spre deosebire de Monte Carlo care așteaptă recompensa finală

### 9. RL_PARAMETERS - Rolul Parametrilor în RL

**Domeniu:** Învățare prin Întărire - Înțelegerea Conceptuală

**Descriere Detaliată:**
Acest tip de întrebare se concentrează pe înțelegerea rolului parametrilor alpha, gamma și epsilon în algoritmii de învățare prin întărire, în special în contextul epsilon-greedy Q-learning.

**Întrebări Tipice:**
- Care este rolul fiecărui parametru?
- Ce se întâmplă dacă setăm un parametru la 0?
- Cum afectează parametrii convergența și performanța?

**Alpha (Learning Rate) - Detaliat:**

**Rol:**
- Controlează viteza de învățare din experiențe noi
- "Câtă încredere avem în informația nouă vs cea veche"

**Setare la 0:**
- V(s) sau Q(s,a) nu se mai actualizează niciodată
- Agentul nu învață nimic din experiență
- Rămâne cu valorile inițiale (de obicei 0)

**Valori mici (ex: 0.1):**
- Învățare lentă, graduală
- Stabilitate mare, volatilitate mică
- Bun pentru medii stochastice (zgomotoase)

**Valori mari (ex: 0.9):**
- Învățare rapidă
- Poate fi instabil, volatil
- Bun pentru medii deterministe

**Gamma (Discount Factor) - Detaliat:**

**Rol:**
- Ponderează importanța recompenselor viitoare
- "Cât de mult ne pasă de viitor vs prezent"

**Setare la 0:**
- Agentul devine complet miop
- Valorează doar recompensa imediată r
- Ignoră complet toate recompensele viitoare
- max_a' Q(s',a') este înmulțit cu 0, deci dispare

**Valori mici (ex: 0.3):**
- Agent cu orizont scurt
- Preferă recompense imediate
- Bun pentru sarcini episodice scurte

**Valori mari (ex: 0.95):**
- Agent cu viziune pe termen lung
- Dispus să sacrifice recompense imediate pentru câștiguri viitoare
- Bun pentru sarcini cu recompense întârziate

**Epsilon (Exploration Rate) - Detaliat:**

**Rol:**
- Balanța între exploatare (folosește ce știi) și explorare (încearcă lucruri noi)
- Implementează strategia epsilon-greedy

**Setare la 0:**
- Agentul devine pur greedy (exploatare 100%)
- Alege întotdeauna acțiunea cu Q maxim
- Risc: poate rămâne blocat într-un optim local
- Nu explorează niciodată alternative potențial mai bune

**Valori mici (ex: 0.1):**
- 10% explorare, 90% exploatare
- Bun după ce am învățat mult
- Fază de "rafinare"

**Valori mari (ex: 0.5):**
- 50% explorare, 50% exploatare
- Bun la început când nu știm nimic
- Fază de "descoperire"

**Decay Strategies:**
În practică, epsilon scade în timp (ex: epsilon = epsilon_0 × 0.995^episode)
- Început: explorare mare pentru a descoperi mediul
- Sfârșit: explorare mică pentru a exploata ce am învățat

### 10. MDP_COMPARISON - Comparație Algoritmi MDP

**Domeniu:** Procese Decizionale Markov - Analiză Comparativă

**Descriere Detaliată:**
Acest tip de întrebare cere comparația între algoritmii Value Iteration și Policy Iteration în termeni de:
- Complexitate computațională
- Viteza de convergență
- Număr de iterații
- Costuri per iterație
- Scenarii în care unul este preferabil

**Comparație Detaliată:**

**Value Iteration:**
- Actualizări asincrone: valorile pot fi folosite imediat ce sunt calculate
- O singură trecere prin toate stările per iterație
- Mai multe iterații până la convergență
- Fiecare iterație este mai rapidă
- Complexitate per iterație: O(S² × A)
- Bun pentru probleme mari unde evaluarea completă e costisitoare

**Policy Iteration:**
- Două faze distincte: evaluare + îmbunătățire
- Evaluare: multiple iterații pentru a calcula V^π precis
- Îmbunătățire: o singură trecere prin stări
- Mai puține iterații externe (schimbări de politică)
- Fiecare iterație externă este mai costisitoare
- Converge mai repede în sensul număr de politici evaluate
- Bun pentru probleme mici/medii unde evaluarea precisă e fezabilă

**În Practică:**
- Value Iteration: mai popular în probleme mari, mai simplu de implementat
- Policy Iteration: poate converge mai rapid în număr de pași
- Modified Policy Iteration: hibrid - evaluare parțială în loc de completă

### 11. STRIPS_ACTION_DEFINITION - Definire Acțiuni STRIPS

**Domeniu:** Planificare Clasică - Stanford Research Institute Problem Solver

**Descriere Detaliată:**
Acest tip de întrebare cere definirea formală a unei acțiuni în limbajul STRIPS, specificând explicit precondițiile, add-list (efectele pozitive) și delete-list (efectele negative).

**Limbajul STRIPS:**

**Reprezentare:**
Fiecare acțiune are trei componente:
1. **Precondiții:** Ce trebuie să fie adevărat înainte de executarea acțiunii
2. **Add-list:** Ce devine adevărat după executarea acțiunii
3. **Delete-list:** Ce devine fals după executarea acțiunii

**Domenii Implementate:**

**Shopping Domain:**
- Acțiune Go(from, to): Mutarea agentului între locații
  - Precondiții: At(agent, from)
  - Add-list: At(agent, to)
  - Delete-list: At(agent, from)
  
- Acțiune Buy(item, store): Cumpărarea unui obiect
  - Precondiții: At(agent, store), Sells(store, item)
  - Add-list: Have(item)
  - Delete-list: (niciunul - nu pierdem nimic prin cumpărare)

**BlocksWorld Domain:**
- Acțiune FromTable(block): Ridicarea unui cub de pe masă
  - Precondiții: OnTable(block), Clear(block), HandEmpty()
  - Add-list: Holding(block)
  - Delete-list: OnTable(block), Clear(block), HandEmpty()
  
- Acțiune Stack(block1, block2): Așezarea unui cub peste altul
  - Precondiții: Holding(block1), Clear(block2)
  - Add-list: On(block1, block2), Clear(block1), HandEmpty()
  - Delete-list: Holding(block1), Clear(block2)

**Container Domain:**
- Acțiune PlaceCap(container): Închiderea unui container
  - Precondiții: Open(container), ¬HasCap(container)
  - Add-list: HasCap(container)
  - Delete-list: Open(container)
  
- Acțiune Insert(item, container): Introducerea obiect în container
  - Precondiții: Open(container), Holding(item)
  - Add-list: Inside(item, container), HandEmpty()
  - Delete-list: Holding(item)

**Randomizare:**
- Domeniul este ales aleator (shopping, blocksworld, container)
- Acțiunea specifică din domeniu este aleasă aleator
- Parametrii acțiunii (blocuri, obiecte, locații) sunt randomizați

### 12. ADL_ACTION_DEFINITION - Definire Acțiuni ADL

**Domeniu:** Planificare Clasică - Action Description Language

**Descriere Detaliată:**
ADL (Action Description Language) este o extensie a limbajului STRIPS care permite expresivitate crescută prin:
- Precondiții complexe (disjuncții, negații, cuantificatori)
- Efecte condiționate (CÂND condiție ATUNCI efect)
- Parametri tipați
- Funcții și fluenți

**Extensii față de STRIPS:**

**Precondiții Complexe:**
- STRIPS: doar conjuncții de predicate pozitive (P AND Q AND R)
- ADL: permite disjuncții (P OR Q), negații (NOT P), cuantificatori (FORALL, EXISTS)

**Efecte Condiționate:**
Formatul: CÂND condiție ATUNCI efect
Exemplu: CÂND Heavy(package) ATUNCI Slow(vehicle)

**Domenii Implementate:**

**Logistics Domain:**
- Acțiune Load(package, vehicle):
  - Precondiții: At(package, location), At(vehicle, location)
  - Add-list: In(package, vehicle)
  - Delete-list: At(package, location)
  - Efect condițional: CÂND Heavy(package) ATUNCI Slow(vehicle)
  
  Interpretare: Dacă pachetul este greu, vehiculul devine lent după încărcare

**Robot Navigation Domain:**
- Acțiune Move(from, to):
  - Precondiții: At(robot, from)
  - Add-list: At(robot, to)
  - Delete-list: At(robot, from)
  - Efect condițional: CÂND Dark(to) ATUNCI NeedLight(robot)
  
  Interpretare: Dacă camera destinație este întunecată, robotul are nevoie de lumină

**Avantaje ADL:**
- Mai expresiv: poate modela probleme mai complexe
- Mai natural: reflectă mai bine realitatea (efecte dependente de context)
- Mai compact: un număr mai mic de acțiuni poate acoperi mai multe scenarii

### 13. PARTIAL_ORDER_PLAN - Planificare cu Ordine Parțială (POP)

**Domeniu:** Planificare Clasică - Algoritmi de planificare

**Descriere Detaliată:**
Acest tip de întrebare cere construirea unui plan incomplet folosind algoritmul Partial Order Planning (POP). Spre deosebire de planificarea cu ordine totală, POP specifică doar constrângerile de ordine necesare între acțiuni, lăsând alte ordini flexibile.

**Concepte Cheie:**

**Plan Incomplet:**
Nu toate acțiunile sunt complet ordonate - avem doar relații de precedență necesare
Exemplu: "A < B" și "C < B" dar nu specificăm relația între A și C

**Componente POP:**
1. **Acțiuni:** Lista de acțiuni din plan (inclusiv Start și Finish)
2. **Orderings:** Constrângeri de precedență (id1 < id2 înseamnă id1 trebuie înainte de id2)
3. **Causal Links:** Legături cauzale (Producer --predicat--> Consumer)

**Algoritmul POP - Pași Detaliați:**

**Inițializare:**
- Creează acțiunea Start cu efecte = starea inițială
- Creează acțiunea Finish cu precondiții = obiectivele
- Plan inițial: {Start, Finish}, Orderings: {Start < Finish}, Links: {}

**Bucla Principală:**
1. **Selectează un obiectiv nerezolvat g:**
   - Alege o precondiție nesatisfăcută din Finish sau din alte acțiuni

2. **Găsește o acțiune A care produce g:**
   - Caută în acțiunile existente sau adaugă o acțiune nouă
   - A trebuie să aibă g în add-list

3. **Adaugă causal link:**
   - Creează link: A --g--> Consumer (acțiunea care necesită g)
   - Înseamnă: A produce g pentru Consumer

4. **Adaugă ordering constraints:**
   - A < Consumer (A trebuie să vină înainte de Consumer)
   - Start < A (A trebuie după Start)
   - A < Finish (A trebuie înainte de Finish)

5. **Rezolvă amenințările (threats):**
   - O amenințare apare când o acțiune C șterge (delete-list) un predicat p
   - care este necesar într-un causal link A --p--> B
   - Rezolvare prin promotion (C < A) sau demotion (B < C)

6. **Repetă până când toate obiectivele sunt rezolvate**

**Exemplu Concret:**

**Problemă:**
- Stare inițială: At(agent, home), Sells(store1, milk)
- Obiective: Have(milk), At(agent, home)

**Plan POP generat:**

Acțiuni:
- 0: Start (produce At(agent, home), Sells(store1, milk))
- 1: Go(home, store1)
- 2: Buy(milk, store1)
- 3: Go(store1, home)
- 999: Finish (necesită Have(milk), At(agent, home))

Orderings:
- 0 < 1 (Start înainte de Go)
- 1 < 2 (Trebuie să ajungem la magazin înainte de a cumpăra)
- 2 < 3 (Trebuie să cumpărăm înainte de a pleca)
- 3 < 999 (Trebuie să ajungem acasă înainte de Finish)

Causal Links:
- 0 --At(agent, home)--> 1 (Start produce poziția pentru Go)
- 1 --At(agent, store1)--> 2 (Go produce poziția pentru Buy)
- 2 --Have(milk)--> 999 (Buy produce obiectivul Have)
- 3 --At(agent, home)--> 999 (Go produce obiectivul At)

**Flexibilitate:**
- Nu am specificat dacă cumpărăm laptele înainte sau după ce facem altceva
- Planul permite paralelizare potențială
- Poate fi transformat în plan cu ordine totală prin adăugarea constrângerilor suplimentare

### 14. PLAN_VALIDATION - Validarea Planurilor

**Domeniu:** Planificare Clasică - Verificare corectitudine

**Descriere Detaliată:**
Acest tip de întrebare prezintă un plan (secvență de acțiuni) și cere verificarea corectitudinii lui: dacă toate precondițiile sunt satisfăcute la momentul executării fiecărei acțiuni și dacă obiectivele sunt atinse la final.

**Metodologie de Validare:**

**Forward Search (Căutare înainte):**
1. **Inițializare:** Stare curentă = Stare inițială
2. **Pentru fiecare acțiune în plan:**
   a. **Verifică precondiții:** Toate predicatele din preconditions trebuie în starea curentă
   b. **Dacă precondiții nesatisfăcute:** Plan invalid - raportează eroarea
   c. **Aplică efecte:**
      - Șterge predicatele din delete-list din starea curentă
      - Adaugă predicatele din add-list la starea curentă
3. **Verifică obiective:** Toate predicatele din goals trebuie în starea finală
4. **Dacă obiective neatinse:** Plan invalid - raportează eroarea

**Tipuri de Erori Detectate:**

**Precondiții Nesatisfăcute:**
Exemplu: Acțiunea Buy(milk, store1) necesită At(agent, store1) dar agentul este la home
Mesaj: "Acțiunea 2 (Buy(milk, store1)) are precondiții nesatisfăcute"

**Ordine Incorectă:**
Exemplu: Încercăm să cumpărăm înainte de a merge la magazin
Detectat prin verificarea precondițiilor - At(agent, store) lipsește

**Obiective Neatinse:**
Exemplu: Planul se termină dar agentul nu are laptele sau nu este acasă
Mesaj: "Obiectivele nu sunt atinse la sfârșitul planului"

**Probleme de Consistență:**
Exemplu: Planul șterge un predicat necesar mai târziu
Detectat prin simularea pas cu pas

**Randomizare:**
- 50% planuri corecte, 50% planuri cu erori
- Erorile sunt introduse deliberat: ordine greșită, acțiuni lipsă
- Variabilitate: diferite tipuri de erori în generări diferite

---

## Cum Funcționează Generatorii

### Arhitectură Comună

Toți generatorii implementează clasa abstractă `QuestionGenerator` cu metoda `generate()`. Factory-ul `QuestionFactory` creează instanțe de generatori bazat pe tipul dorit.

### Proces de Generare

**Pasul 1: Randomizare Parametri**
- Fiecare generator randomizează parametrii problemei
- Exemplu: dimensiune tablă, număr obiecte, valori, configurații
- Folosește biblioteca `random` pentru selecții aleatoare

**Pasul 2: Crearea Instanței Problemei**
- Generează structura de date specifică problemei
- Exemplu: `NashGame`, `CspInstance`, `GridWorld`, `PlanningProblem`
- Populează cu date randomizate

**Pasul 3: Rezolvarea Problemei**
- Apelează solver-ul corespunzător prin `SolverFactory`
- Obține răspunsul corect și explicația
- Exemplu: pentru Nash, găsește echilibrele; pentru CSP, găsește asignările

**Pasul 4: Formatarea Întrebării**
- Creează text descriptiv pentru student
- Include toate informațiile necesare pentru rezolvare
- Format clar, structurat, ușor de citit

**Pasul 5: Returnarea Obiectului Question**
- Obiect complet cu: id, titlu, text, tip, topic, dificultate
- Include răspuns corect, explicație, metadata
- Gata pentru afișare în GUI sau export PDF

### Strategii de Randomizare

**Variație Parametri Numerici:**
- Dimensiuni: `random.randint(min, max)`
- Valori: `random.uniform(min, max)` pentru float
- Alegeri: `random.choice([listă opțiuni])`

**Variație Structurală:**
- Număr componente: variabil (ex: 3-5 variabile CSP)
- Configurații: diferite layout-uri de grid, grafuri
- Combinații: diferite seturi de obiecte, acțiuni

**Asigurare Solvabilitate:**
- Problemele generate sunt întotdeauna solvabile
- Pentru CSP: domenii suficient de mari
- Pentru planning: există întotdeauna un plan valid

---

## Cum Funcționează Solverele

### Arhitectură Comună

Toți solverii implementează interfața `Solver` cu metoda abstractă `solve(data: Dict) -> Dict`. Factory-ul `SolverFactory` creează instanțe bazat pe tipul problemei.

### Solvere Specifice

**StrategySolver:**
- Primește numele problemei (n-queens, hanoi, etc.)
- Returnează strategia recomandată dintr-o mapare predefinită
- Generează explicație despre de ce strategia este potrivită

**NashSolver:**
- Primește obiectul `NashGame`
- Iterează prin toate cele 4 combinații posibile (2×2)
- Pentru fiecare combinație, verifică dacă este echilibru Nash
- Returnează lista de echilibre găsite

**CspSolver:**
- Implementează Backtracking recursiv
- Folosește MRV pentru selectarea variabilei următoare
- Aplică Forward Checking după fiecare asignare
- Returnează prima soluție validă găsită

**MinimaxAlphaBetaSolver:**
- Implementează algoritm Minimax recursiv cu Alpha-Beta pruning
- Contorizează frunzele evaluate (nu și cele tăiate)
- Returnează valoarea rădăcinii și numărul de frunze

**ValueIterationSolver:**
- Implementează ecuația Bellman iterativ
- Iterează până la convergență (delta < epsilon)
- Extrage politica optimă din valorile finale
- Returnează dicționar de utilități și politică

**PolicyIterationSolver:**
- Alternează între evaluarea politicii și îmbunătățirea ei
- Evaluare: rezolvă sistem de ecuații liniare pentru V^π
- Îmbunătățire: calculează politica greedy bazată pe V^π
- Returnează când politica nu se mai schimbă

**QLearningSolver:**
- Aplică formula Q-learning pentru fiecare tranziție
- Actualizează Q(s,a) folosind TD-error
- Returnează tabel Q final și politică optimă

**TDLearningSolver:**
- Aplică formula TD(0) pentru fiecare tranziție
- Actualizează V(s) folosind TD-error
- Returnează valorile stărilor și TD-errors

**StripsActionFormatterSolver:**
- Primește obiect `Action`
- Formatează în limbaj STRIPS: precondiții, add-list, delete-list
- Returnează text formatat pentru afișare

**AdlActionFormatterSolver:**
- Similar cu STRIPS dar include efecte condiționate
- Formatează: CÂND condiție ATUNCI efect
- Returnează text formatat ADL

**PartialOrderPlanningSolver:**
- Implementează algoritm POP simplificat
- Creează plan cu Start, acțiuni necesare, Finish
- Generează orderings și causal links
- Returnează obiect `PartialOrderPlan`

**ForwardSearchPlanningSolver:**
- Simulează executarea planului pas cu pas
- Verifică precondiții la fiecare pas
- Verifică obiective la final
- Returnează validitate și lista de erori

---

## Cum Funcționează Evaluatorii

### Arhitectură Comună

Toți evaluatorii implementează clasa abstractă `AnswerEvaluator` cu metoda `evaluate(question, user_answer) -> EvaluationResult`. Factory-ul `EvaluatorFactory` creează instanțe bazat pe tipul întrebării.

### Procesare Text

**TextProcessor:**
Toate evaluatorile folosesc `TextProcessor` pentru normalizare:
- Convertire la lowercase
- Eliminare diacritice (ă→a, ș→s, etc.)
- Lematizare simplă (plurale la singular, verbe la infinitiv)
- Eliminare cuvinte stop (de, la, cu, etc.)

### Metode de Scoring

**Keyword Scoring:**
- Compară cuvinte cheie din răspunsul corect cu răspunsul userului
- Calculează procentul de overlap
- Folosit de: StrategyEvaluator, NashEvaluator

**Numerical Comparison:**
- Extrage valori numerice din răspuns
- Compară cu toleranță (ex: ±0.01 pentru utilități MDP)
- Folosit de: MDPEvaluator, RLEvaluator, MinimaxEvaluator

**Structural Comparison:**
- Verifică prezența componentelor structurale
- Exemplu: politică + valori pentru MDP
- Scoring gradual: componente parțiale = punctaj parțial

**Semantic Comparison:**
- Verifică keywords semantici (nu doar string exact)
- Exemplu: "precondițiile" sau "precondiții" sau "precond"
- Folosit de: PlanningEvaluator

### Evaluatori Specifici

**StrategyEvaluator:**
- Normalizează ambele răspunsuri
- Calculează keyword score
- Feedback: procentaj + strategia corectă

**NashEvaluator:**
- Similar cu StrategyEvaluator
- Căută mențiuni de echilibre Nash
- Poate detecta "nu există" vs echilibre specifice

**CspEvaluator:**
- Verifică fiecare asignare variabilă = valoare
- Punctaj proporțional cu numărul de asignări corecte
- Feedback detaliat per variabilă

**MinimaxEvaluator:**
- Extrage valoare rădăcină și număr frunze din răspuns
- Compară cu valorile corecte
- 50% pentru valoare corectă, 50% pentru număr frunze

**MDPEvaluator:**
- Verifică prezența valorilor utilităților
- Verifică prezența politicii
- Răspuns parțial (doar valori) = 25%, complet = 100%
- Comparare numerică cu toleranță pentru valori

**RLEvaluator:**
- Verifică mențiuni de Q-values sau V-values
- Verifică mențiuni de politică
- Verifică înțelegerea parametrilor (alpha, gamma, epsilon)

**PlanningEvaluator:**
- Pentru STRIPS/ADL: verifică precondiții (30%), add-list (35%), delete-list (35%)
- Pentru POP: verifică acțiuni (20%), orderings (30%), causal links (30%), goals (20%)
- Pentru validare: verifică identificare corectă valid/invalid (50%), descriere erori (50%)
- Keyword matching flexibil pentru variații de exprimare

---

## Q&A Service - Sistem Interactiv

### Arhitectură Q&A

**Componente:**
1. **ProblemParser** - Extrage tip și date din limbaj natural
2. **SolverFactory** - Creează solver-ul potrivit
3. **QAService** - Orchestrează parsing → solving → formatting

### Procesul Q&A - Detaliat

**Pasul 1: Parsing**
Utilizatorul introduce întrebare în limbaj natural:
"Găsește echilibrul Nash într-un joc cu payoff-uri (3,2) și (1,4) pe primul rând..."

**ProblemParser:**
- Calculează confidence score pentru fiecare extractor
- Exemplu: `NashGameExtractor` detectează "nash", "echilibru", "payoff" → confidence 0.9
- Selectează extractor-ul cu cel mai mare scor
- Extractor-ul parseaza textul și extrage date structurate

**Pasul 2: Verificare Ambiguitate**
- Verifică dacă multiple tipuri au confidence mare (>0.3)
- Dacă da: respinge cu "Nu știu să răspund acum"
- Verifică keywords din categorii diferite (nash + csp, mdp + nash, etc.)
- Excepție: STRIPS și ADL pot coexista (același extractor)

**Pasul 3: Rezolvare**
- Obține solver prin `SolverFactory.get_solver(question_type)`
- Apelează `solver.solve(parsed_data)`
- Pentru planning: returnează guidance, nu soluție concretă
- Pentru alte tipuri: returnează soluție calculată

**Pasul 4: Formatare Răspuns**
- Determină tipul întrebării
- Apelează metoda de formatare corespunzătoare:
  - `_format_nash_response()` - formatează echilibre Nash
  - `_format_csp_response()` - formatează asignări CSP
  - `_format_value_iteration_response()` - formatează utilități + politică
  - etc.
- Creează obiect `QAResponse` cu:
  - success: True/False
  - detected_type: tip detectat
  - confidence: scor de încredere
  - extracted_params: parametri extrași (formatați)
  - solution: soluția formatată
  - explanation: explicație despre metoda de rezolvare

**Pasul 5: Afișare în GUI**
- Panel Q&A afișează răspunsul formatat
- Include: tip detectat, confidence, parametri, soluție, explicație
- Design clar, ușor de citit

### Extractori Implementați

**NashGameExtractor:**
- Keywords: nash, equilibrium, game, payoff, player, strategy
- Extrage: dimensiune matrică, strategii, payoff-uri
- Pattern matching pentru tupluri de tip (x, y)

**CspExtractor:**
- Keywords: csp, constraint, variable, domain, assignment
- Extrage: variabile, domenii, constrângeri, asignări parțiale
- Pattern matching pentru relații (X != Y)

**MinimaxExtractor:**
- Keywords: minimax, alpha-beta, pruning, tree, max, min
- Extrage: structura arborelui, valori frunze
- Reconstruiește arbore din descriere textuală

**StrategyExtractor:**
- Keywords: strategy, n-queen, hanoi, coloring, knight
- Detectează problema specifică menționată
- Extrage parametri relevanți (dimensiune, număr discuri, etc.)

**MDPExtractor:**
- Keywords: mdp, markov, value iteration, policy iteration, bellman, grid
- Extrage: dimensiune grid, recompense, discount factor, stări terminale
- Pattern matching pentru coordonate grid

**RLExtractor:**
- Keywords: q-learning, td-learning, alpha, gamma, epsilon, transition
- Extrage: tranziții (s, a, s', r), parametri RL
- Pattern matching pentru Q-values: Q((0,0), right) = 0.5

**StripsAdlExtractor:**
- Keywords: strips, adl, operație, precondiție, add-list, delete-list
- Extrage: domeniu (shopping, blocksworld), nume acțiune
- Detectează ADL prin "condițional", "când", "atunci"

**PopExtractor:**
- Keywords: partial order, pop, plan incomplet, ordine parțială, cauzal
- Extrage: domeniu, predicate menționate, stare inițială, obiective
- Pattern matching pentru predicate: At(agent, home)

**PlanValidationExtractor:**
- Keywords: verificați, validate, corect, plan, acțiuni
- Extrage: secvență acțiuni menționate
- Pattern matching pentru acțiuni: Go(home, store)

### Dezambiguizare

**Strategii de Dezambiguizare:**

**Verificare Keywords Explicit:**
- "alpha-beta" (minimax) vs "alpha" (RL parameter)
- Minimax necesită "alpha-beta" împreună sau "alpha beta"
- RL necesită "alpha =", "gamma =", "learning rate"

**Verificare Payoff Patterns:**
- Nash necesită pattern (x, y) pentru payoff-uri
- Evită false positive cu alte coordonate

**Verificare Categorii Menționate:**
- Contorizează categorii distincte: nash, csp, mdp, rl, planning
- Dacă 2+ categorii detectate: reject
- Excepție: STRIPS + ADL OK (același domeniu)

**Verificare Conectori:**
- "folosind", "cu", "prin" pot indica multiple tipuri
- Exemplu: "CSP folosind Nash" → reject (ambiguu)

---

## Interfața Grafică (GUI)

### Fereastră Principală

**Layout:**
- Stânga: Panel configurare + generare întrebări
- Dreapta: Panel afișare întrebări + evaluare răspunsuri
- Jos: Panel Q&A interactiv

**Panel Configurare:**

**Număr Întrebări:**
- SpinBox: selectează între 1 și 20 întrebări
- Default: 4 întrebări

**Checkboxes Tipuri:**
13 checkboxes pentru fiecare tip de întrebare:
- Strategie pentru problemă
- Echilibru Nash
- CSP cu Backtracking
- MinMax Alpha-Beta
- Value Iteration (MDP)
- Policy Iteration (MDP)
- Q-learning (RL)
- TD-learning (RL)
- Parametri RL
- Definire Acțiune STRIPS
- Definire Acțiune ADL
- Partial Order Planning
- Validare Plan

Toate bifate by default pentru variabilitate maximă

**Buton Generare:**
- Apasă pentru a genera întrebări noi
- Culege tipurile selectate
- Generează numărul specificat de întrebări
- Randomizează ordinea

**Panel Afișare:**

**Listă Întrebări:**
- Afișează toate întrebările generate
- Format: "ID. Titlu"
- Selecție pentru a vedea detalii

**Detalii Întrebare:**
- Titlu complet
- Text întrebă (poate fi lung, scrollable)
- Topic și dificultate
- Câmp text pentru răspuns utilizator

**Buton Verificare:**
- Apasă după introducerea răspunsului
- Evaluează răspunsul folosind evaluatorul corespunzător
- Afișează scor și feedback detaliat

**Butoane Export PDF:**
- "PDF per întrebare" - generează PDF separat pentru fiecare
- "PDF test complet" - toate întrebările într-un singur PDF
- "PDF cu feedback" - include răspunsuri și scoruri

### Panel Q&A Interactiv

**Funcționalitate:**
- Permite introducerea întrebărilor în limbaj natural
- Suportă română și engleză
- Detectează automat tipul problemei
- Returnează soluție sau guidance

**Interfață:**
- Câmp text mare pentru întrebare
- Buton "Întreabă"
- Zonă afișare răspuns cu:
  - Tip detectat
  - Confidence score
  - Parametri extrași
  - Soluție/Guidance
  - Explicație metodă

**Exemple Întrebări Acceptate:**
- "Găsește echilibrul Nash pentru jocul cu payoff-uri..."
- "Rezolvă CSP-ul cu variabile A, B, C..."
- "Descrie operația Go(home, store) în STRIPS"
- "Construiește un plan POP pentru shopping..."

---

## Serviciul PDF

### Generare Documente

**PDF per Întrebare:**
- Header: SmarTest - Întrebare {id}
- Titlu întrebării
- Text întrebării
- Spațiu gol pentru răspuns
- Footer: pagină

**PDF Test Complet:**
- Pagină titlu: "Test Inteligență Artificială"
- Număr total întrebări
- Pentru fiecare întrebare:
  - Număr întrebare
  - Titlu
  - Text
  - Spațiu răspuns
- Numerotare pagini

**PDF Feedback:**
- Include tot de la test complet PLUS:
- Răspunsul utilizatorului
- Scor obținut
- Feedback detaliat
- Răspuns corect
- Explicație

### Formatare

**Font:**
- Titluri: Bold, 14pt
- Text normal: Regular, 11pt
- Monospace pentru cod/formule

**Margini:**
- 2.5cm pe toate laturile
- Spațiere inter-paragraf

**Culori:**
- Header/Footer: gri
- Scor >80%: verde
- Scor 50-80%: portocaliu
- Scor <50%: roșu

---

## Procesare Text

### TextProcessor

**Funcții:**

**normalize(text):**
- Lowercase
- Eliminare diacritice
- Stemming simplu (plurale → singular)
- Eliminare punctuație
- Eliminare cuvinte stop

**keyword_score(correct, user):**
- Extrage cuvinte cheie din ambele texte
- Calculează overlap (cuvinte comune / cuvinte corecte)
- Returnează procentaj 0-100

**extract_numbers(text):**
- Regex pentru identificare numere
- Suportă întregi, float, negative
- Returnează listă numere găsite

**extract_predicates(text):**
- Regex pentru pattern Predicate(params)
- Exemplu: At(agent, home)
- Returnează listă predicate găsite

---

## Instalare și Rulare

### Cerințe

**Python:** 3.10 sau mai nou
**Sistem Operare:** Windows, macOS, Linux

### Instalare

**Pas 1: Clonare/Descărcare**
```
Descarcă sau clonează repository-ul în directorul dorit
```

**Pas 2: Virtual Environment (recomandat)**
```
Creează un mediu virtual Python izolat
Activează mediul virtual
```

**Pas 3: Instalare Dependințe**
```
Folosește pip pentru a instala toate pachetele necesare
Fișierul requirements.txt conține lista completă
```

**Dependințe Principale:**
- PySide6: Framework GUI (Qt6 pentru Python)
- PyPDF2: Manipulare fișiere PDF
- ReportLab: Generare PDF-uri
- Pillow: Procesare imagini (pentru PDF)

### Rulare

**Rulare Normală:**
```
Execută main.py pentru a lansa aplicația
Se deschide fereastra GUI
```

**Rulare Teste:**
```
Execută fișierele de test din directorul tests/
test_mdp_rl.py - teste MDP și RL
test_strips_adl_pop.py - teste Planning
```

---

## Extensibilitate

### Adăugare Tip Nou de Întrebare

**Pasul 1: Model de Date**
Adaugă enum în `QuestionType` și eventual dataclass specific în `models.py`

**Pasul 2: Generator**
Creează clasă nouă care extinde `QuestionGenerator` în `generators.py`
Implementează metoda `generate()` cu randomizare

**Pasul 3: Solver**
Creează clasă nouă care implementează `Solver` în `solvers.py`
Implementează metoda `solve()` cu algoritmul de rezolvare

**Pasul 4: Evaluator**
Creează clasă nouă care extinde `AnswerEvaluator` în `evaluators.py`
Implementează metoda `evaluate()` cu logică de scoring

**Pasul 5: Parser (pentru Q&A)**
Creează clasă nouă care implementează `ProblemExtractor` în `problem_parser.py`
Implementează `can_extract()` și `extract()`

**Pasul 6: Integrare**
Actualizează factory-urile pentru a include noile clase
Adaugă checkbox în GUI
Adaugă metodă de formatare în QAService

**Pasul 7: Teste**
Creează teste pentru generator, solver, evaluator, parser
Verifică randomizare, corectitudine, edge cases

### Pattern-uri Folosite

Arhitectura permite adăugarea de funcționalități noi fără modificarea codului existent (Open-Closed Principle). Factory-urile și interfețele abstracte fac extensia naturală și sigură.

---

## Concluzie

SmarTest este o aplicație comprehensivă și extensibilă pentru generarea, rezolvarea și evaluarea automată a problemelor de Inteligență Artificială. Acoperind 13 domenii majore, de la algoritmi de căutare la planificare clasică, aplicația oferă un instrument complet pentru pregătirea examenelor și înțelegerea profundă a conceptelor AI.

Arhitectura modulară, bazată pe pattern-uri de design solide, permite extinderea ușoară cu noi tipuri de probleme și algoritmi. Sistemul de randomizare asigură variabilitate infinită, iar evaluarea semantică oferă feedback detaliat și util pentru învățare.
