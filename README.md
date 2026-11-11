# Zadanie 1
Zadanie zawiera zarówno cały kod doświadczenia (ModelComparison.ipynb), jaki i dokumentację (.md). Dokumentacja działa w preview githuba.
Przy pobieraniu dokumentacji trzeba również ściągnąć zdjęcia do tego samego folderu.

# Zadanie 2
Zadanie zawiera kod w formacie .ipynb oraz notatkę z analizowanych po wykładzie zagadnień

# Zadanie 3

### Problem Badawczy
Jak zmiana standardowego threshold:
```math
\hat{y} = \begin{cases}
-1 & f(x; \theta) < 0 \\
1 & \text{w przeciwnym razie}
\end{cases}
```
na classical head: $\hat{y} = sign(wf + b)$

wpłynie na accuracy modelu przy tej samej liczbie epok. Czyli dodajemy dwa parametry, które będą modyfikowały output modelu.

Oba modele buduje na identycznych:
- parametrach początkowych ($\theta$ initial)
- learning rate (${}lr{}$)
- architekturach ansatz'u
### Hipoteza
$l = (f - y)^{2}$
to wzór na MSE, którego będę używał do trenowania modelu (minimalizowania funkcji straty). $y \in \lbrace -1, 1 \rbrace{}$. Gdyż ta wartość definiuje klasę, to by minimalizować funkcję straty, ${}f{}$ również będzie zbliżać się do tych wartości. Ponieważ (dla standard threshold) $f = \braket{Z_{0}} \in [-1, 1]$ to parametry $\theta$ będą musiały być skrajne. To może być szkodliwe dla gradientu, gdyż analizujemy kąty ${}\theta{}$ dla których ${}f{}$ ma wartości skrajne, więc jej pochodna czyli właśnie gradient, jest wypłaszczony.

Użycie dodatkowych parametrów ${}w{}$ i ${}b{}$ pozwala na wyliczenie:
```math
wf + b
```
Z tego powodu $f = \braket{Z_{0}}$ nie musi już osiągać wartości skrajnych, by minimalizować funkcję straty. Którą teraz wyrażamy wzorem:
```math
l = ((wf + b) - y)^{2}
```
Ważna uwaga - do samej klasyfikacji używamy: $\hat{y} = sign(wf + b)$.

Model ze standard threshold będzie gorzej dopasowany w porównaniu z modelem z classical head, gdzie ${}\theta{}$ nie musi osiągać skrajnych wartości.

Wnioskując z powyższych zależności, uważam, że skorzystanie z classical overhead da wyższe o przynajmniej 3 punkty procentowe accuracy od standardowego threshold przy tej samej liczbie epok.
