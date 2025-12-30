# Fix-uri: Randomizare și Evaluare

## Probleme Identificate și Rezolvate

### ❌ Problema 1: Generatori Deterministici
**Simptom**: Fiecare tip de întrebare genera mereu exact aceeași întrebare, fără variație.

**Cauză**: Generatorii foloseau valori hardcodate pentru toate parametrii:
- Grid world mereu 3x4 cu aceleași recompense
- Q-learning mereu cu aceleași tranziții
- TD-learning mereu cu aceleași secvențe

**Soluție ✅**: 

#### Value Iteration Generator
- **Grid dimensions**: Randomizat între 2-4 rânduri × 3-5 coloane
- **Living cost**: Randomizat între -0.04, -0.02, -0.1
- **Goal reward**: Randomizat între 1.0, 2.0, 5.0
- **Penalty reward**: Randomizat între -1.0, -2.0, -5.0
- **Walls**: 0-2 pereți plasați aleator
- **Transition probs**: Intended între 0.7-0.9, perpendicular calculat
- **Discount factor**: 0.8-0.85 (easy), 0.9-0.92 (medium), 0.95-0.99 (hard)
- **Target state**: Aleator din stările non-terminale, non-wall

#### Q-Learning Generator
- **Număr tranziții**: 2-3 (easy), 3-5 (medium), 5-8 (hard)
- **Stări**: Generate aleator pe grid 3×4
- **Acțiuni**: Alese aleator din [up, down, left, right]
- **Recompense**: -0.04, -0.02, -0.1, 0.0 (intermediare), 1.0-5.0 (final)
- **Parametri**: α ∈ {0.1, 0.2, 0.3}, γ ∈ {0.8, 0.9, 0.95}, ε ∈ {0.1, 0.2, 0.3}

#### TD-Learning Generator
- **Număr tranziții**: 2-3 (easy), 3-5 (medium/hard)
- **Stări**: Generate aleator similar cu Q-learning
- **Recompense**: -0.04, -0.02 (intermediare), 1.0-2.0 (final)
- **Parametri**: α ∈ {0.1, 0.2, 0.3}, γ ∈ {0.8, 0.9, 0.95}

**Rezultat**:
```
Value Iteration - 3 questions:
1. Grid dimensions: 4x4
2. Grid dimensions: 2x3
3. Grid dimensions: 2x4

Q-learning - 3 questions:
1. Parameters: α=0.1, γ=0.9, ε=0.2
2. Parameters: α=0.3, γ=0.95, ε=0.3
3. Parameters: α=0.3, γ=0.95, ε=0.1
```

---

### ❌ Problema 2: Evaluator Prea Permisiv
**Simptom**: Răspunsul parțial "V((2, 0)) = -0.04" primea 100%, deși lipsea politica.

**Exemplu**:
```
Correct answer: V((2, 0)) = -0.04, politica: up
User answer: V((2, 0)) = -0.04
Score: 100.0% ❌ INCORECT - ar trebui penalizat
```

**Cauză**: `MDPEvaluator` compara doar valorile numerice și ignora absența politicii.

**Soluție ✅**: 

Evaluatorul verifică acum ce este așteptat și ce este furnizat:

1. **Detectare ce lipsește**:
   ```python
   correct_has_value = "V(" in question.correct_answer
   correct_has_policy = "politica" in question.correct_answer
   user_has_value = "V(" in user_answer or any(char.isdigit() for char in user_answer)
   user_has_policy = "politica" in user_answer or action_words in user_answer
   ```

2. **Penalizare pentru răspuns parțial**:
   - Dacă se așteaptă ambele (valoare + politică):
     - Doar valoare furnizată: `score = value_score * 0.5` (maxim 50%)
     - Doar politică furnizată: `score = policy_score * 0.5` (maxim 50%)
     - Ambele furnizate: `score = 0.6 * value_score + 0.4 * policy_score`

3. **Feedback explicit**:
   ```
   "Ai furnizat doar valoarea, lipseste politica!"
   "Ai furnizat doar politica, lipseste valoarea!"
   ```

**Rezultat**:
```
Correct answer: V((1, 2)) = -0.10, politica: up
Partial answer: V((1, 0)) = 0.50
Score: 25.0% ✓ CORECT - penalizat pentru răspuns parțial
Feedback: Ai furnizat doar valoarea, lipseste politica!
```

---

## Îmbunătățiri Secundare

### 1. Sortare Deterministică în Output
- Grid-urile afișează acum stările sortate pentru consistență
- Ușurează compararea între întrebări diferite

### 2. Validare Stări în Value Iteration
- Target state ales doar din stări valide (non-terminal, non-wall)
- Evită întrebări imposibile sau nesemnificative

### 3. Parametri Realistici
- Toate valorile randomizate sunt în range-uri realiste din practică
- α ∈ [0.1, 0.3] - learning rate tipic
- γ ∈ [0.8, 0.95] - discount factor standard
- ε ∈ [0.1, 0.3] - exploration rate moderat

---

## Testing

### Test Randomizare
```python
# Generate 3 Value Iteration questions
for i in range(3):
    q = app.generate_questions([QuestionType.VALUE_ITERATION], 1)[0]
    # Each has different dimensions, rewards, walls, etc.
```

**Rezultat**: ✅ Toate 3 întrebările sunt diferite

### Test Evaluare Parțială
```python
q = app.generate_questions([QuestionType.VALUE_ITERATION], 1)[0]
# Correct: "V((1, 2)) = -0.10, politica: up"

result = app.evaluate_answer(q, "V((1, 0)) = 0.50")  # Partial answer
# Score: 25.0% (penalizat pentru lipsa politicii)
```

**Rezultat**: ✅ Răspuns parțial penalizat corect

### Test Suite
```bash
python3 tests/test_mdp_rl.py
```

**Rezultat**: ✅ All 11 tests passed

---

## Impact

### Înainte:
- ❌ Același grid 3×4 mereu
- ❌ Aceleași tranziții Q-learning
- ❌ 100% pentru răspuns parțial
- ❌ Lipsă varietate în antrenament

### După:
- ✅ Grid-uri diferite (2-4 × 3-5)
- ✅ Tranziții randomizate
- ✅ Evaluare corectă (50% max pentru parțial)
- ✅ Varietate infinită de întrebări
- ✅ Feedback explicit pentru răspunsuri incomplete

---

## Exemplu Complet

### Generare 3 Întrebări Value Iteration:

**Întrebare 1**:
```
Grid World MDP de dimensiune 4x4
γ = 0.9
(0,3): 2.0 (TERMINAL) - goal
(2,1): -1.0 (TERMINAL) - penalty
(1,1): PERETE
Living cost: -0.04
Target: (3,2)
```

**Întrebare 2**:
```
Grid World MDP de dimensiune 2x3
γ = 0.92
(0,2): 5.0 (TERMINAL) - goal
(1,1): -2.0 (TERMINAL) - penalty
Living cost: -0.02
Target: (0,1)
```

**Întrebare 3**:
```
Grid World MDP de dimensiune 3x5
γ = 0.85
(2,4): 1.0 (TERMINAL) - goal
Living cost: -0.1
No walls
Target: (1,2)
```

### Evaluare Răspunsuri:

| Răspuns | Score | Feedback |
|---------|-------|----------|
| `V((1,2)) = -0.10, politica: up` | 100% | Complet corect ✓ |
| `V((1,2)) = -0.10` | 50% | Lipsește politica! |
| `politica: up` | 50% | Lipsește valoarea! |
| `V((1,2)) = -0.15, politica: up` | 60% | Valoare greșită, politică OK |
| `(răspuns gol)` | 0% | Răspuns incomplet |

---

## Fișiere Modificate

1. **`smartest/core/generators.py`**
   - `ValueIterationGenerator._create_grid_world()` - randomizare completă
   - `ValueIterationGenerator.generate()` - target state aleator
   - `QLearningGenerator._generate_transitions()` - tranziții random
   - `QLearningGenerator.generate()` - parametri random
   - `TDLearningGenerator.generate()` - tranziții și parametri random

2. **`smartest/core/evaluators.py`**
   - `MDPEvaluator.evaluate()` - logică completă de penalizare pentru răspunsuri parțiale

---

## Concluzii

✅ **Problema 1 rezolvată**: Generatorii creează acum întrebări variate  
✅ **Problema 2 rezolvată**: Evaluatorul penalizează corect răspunsurile parțiale  
✅ **Toate testele trec**: 11/11 tests passed  
✅ **0 erori de linting**  
✅ **Varietate infinită**: Studenții nu mai pot memoriza întrebările

**Ready for production!** 🚀

