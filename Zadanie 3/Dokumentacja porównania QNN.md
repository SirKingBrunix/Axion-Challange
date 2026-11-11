# Dokumentacja QNN Comparison

### Problem Badawczy
Jak zmiana standardowego threshold
```math
\hat{y} = \begin{cases} -1 & f(x; \theta) < 0\\ 1 & \text{w przeciwnym razie} \end{cases}
```

na classical head

```math
\hat{y} = \text{sign}(wf + b)
```
wpłynie na accuracy modelu przy tej samej liczbie epok. Czyli dodajemy dwa parametry, które będą modyfikowały output modelu.

Oba modele buduje na identycznych:
- parametrach początkowych ($\theta$ initial)
- learning rate (${}lr{}$)
- architekturach ansatz'u
### Hipoteza
$l = (f - y)^{2}$ to wzór na MSE, którego będę używał do trenowania modelu (minimalizowania funkcji straty). $y \in \lbrace -1, 1 \rbrace{}$. Gdyż ta wartość definiuje klasę, to by minimalizować funkcję straty, ${}f{}$ również będzie zbliżać się do tych wartości. Ponieważ (dla standard threshold) $f = \braket{Z_{0}} \in [-1, 1]$ to parametry $\theta$ będą musiały być skrajne. To może być szkodliwe dla gradientu, gdyż analizujemy kąty ${}\theta{}$ dla których ${}f{}$ ma wartości skrajne, więc jej pochodna czyli właśnie gradient, jest wypłaszczony.

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
***
#### Plan doświadczenia:
1. Wybrać odpowiedni dataset z dwoma features i klasyfikacją binarną
2. Stworzyć odpowiednie datasets (wystarczający noise, by model nie miał ani 50% dokładności, ani 100% dokładności)
3. Stworzyć QNN z standard threshold
4. Stworzyć QNN z classical overhead
5. Wytrenować i przetestować modele na danych
6. Porównać wyniki i konfiguracje
7. Sformułować wnioski i obserwacje
***
#### Dataset i EDA
Zdecydowałem się do doświadczenie stworzyć bazy z dwoma features za pomocą funkcji make_circles i make_moons. Stworzę jedną bazę danych za pomocą każdej funkcji i porównam modele na tych dwóch datasetach.

Ważne by datasety były tak stworzone, by model miał szansę na poprawę (za pomocą wprowadzenia classical overhead), ale nie był losowy, by można było wywnioskować realną poprawę (nie wynikającą ze "szczęścia" modelu).

Do zrobienia takiego rozgraniczenia wykorzystamy algorytm kNN, który został omówiony w zadaniu 1. Na podstawie analizy jego accuracy dobierzemy odpowiedni noise.
##### Skalowanie
By dane dawały odpowiednie stany w modelu QNN, muszą one być odpowiednio przeskalowane (ponieważ funkcje trygonometryczne są okresowe - poruszamy się na sferze Blocha). Wybieramy zatem zakres $[-\pi, \pi]$ i stosujemy skalowanie izotropowe, by nie zmieniać kształtu danych.

Funkcja make_moon i make_circles tworzy dane w idealnej proporcji bez powtórek. To oznacza, że przeskalowane dane są zbalansowane klasami.

Na podstawie wyników algorytmu kNN na bazach danych stworzonych za pomocą:
`X, y = make_moons(n_samples = 400, noise = noise, random_state = 42)`,
gdzie noise $\in \lbrace 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6 \rbrace$

Wybieramy noise = ${}0.4{}$, dla którego accuracy = ${}0.79{}$

Dla baz danych `make_circles`:
`X, y = make_circles(n_samples = 400, noise = noise, factor = 0.5, random_state = 42)`
gdzie noise $\in \lbrace 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6 \rbrace$

Wybieramy noise = ${}0.3{}$, dla którego accuracy = ${}0.75{}$

Finalnie datasety:
![[Pasted image 20261111205320.png]](Images/Pasted%20image%2020261111205320.png)

Dzielimy datasety na:
- X_train_c i y_train_c (80%) - zbiór do trenowania na dataset circles
- X_test_c i y_test_c (20%) - zbiór do testowania na dataset circles
- X_train_m i y_train_m (80%) - zbiór do trenowania na dataset moons
- X_test_m i y_test_m (20%) - zbiór do testowania na dataset moons
#### QNN threshold i overhead
Do obu sieci użyję tego samego obwodu, aby różniły one jedynie obecnością classical overhead:
![[Pasted image 20261111194020.png]](Images/Pasted%20image%2020261111194020.png)

Do obu funkcji trenujących używam parametrów:
- num_epochs = ${}50{}$
- batch_size = ${}32{}$
- lr = ${}0.1{}$
- seed = ${}0{}$
#### Trening
W obu funkcjach trenujących zachowuje dane, by zobaczyć jak przebiegało uczenie.
`history = {"loss": [], "train_acc": [], "test_acc": [], "grad_norm": [], "grad_norm_full": [], "mean_abs_f_train": []}`

1. Loss - wynik funkcji straty
2. train_acc - accuracy na części treningowej
3. test_acc - accuracy na części testowej
4. grad_norm - średnia norma gradientu na batch
5. grad_norm_full - średnia norma gradientu na tym samym datasecie
6. mean_abs_f_train - średnia wartość bezwzględna z ${}f(\theta){}$

Ponadto w funkcji trenującej z overhead zachowuję również parametry w i b.
#### Wyniki i wnioski

![[Pasted image 20261111230622.png]](Images/Pasted%20image%2020261111230622.png)

![[Pasted image 20261111230630.png]](Images/Pasted%20image%2020261111230630.png)

![[Pasted image 20261111230654.png]](Images/Pasted%20image%2020261111230654.png)

![[Pasted image 20261111230636.png]](Images/Pasted%20image%2020261111230636.png)


Powyższe wyniki z pewnością nie dają jasnych i klarownych wniosków, które pozwoliłyby obalić lub potwierdzić hipotezę. Nie należy, więc próbować dopasowywać rezultatów do hipotezy, a zauważyć pojawiające się zależności.

Hipoteza w swojej pierwotnej formie nie została potwierdzona przy pomocy tego doświadczenia. Przy analizie wykresu "Accuracy Circles" różnica w wartościach jest widoczna, jednak wykres dla drugich danych obrazuje jak zbliżone do siebie są wartości dla obu modeli.

Jeszcze większe różnice w wynikach widać na wykresach mean |f(θ)| — przy circles wartość dla overhead jest niemal od początku stała, podczas gdy dla moons jest bardzo podobna do wartości modelu threshold, bez wyraźnej różnicy między wariantami.

Pojawia się podhipoteza, która sugerowałaby, że dla małego ${}w{}$ gradient też jest mały (poprzez czynnik ${}w{}$ w regule łańcuchowej), więc proces uczenia zostaje zahamowany. Jednak wykres ${}w{}$ i ${}b{}$ w czasie obala tę podhipotezę, gdyż ${}w{}$ trzyma się blisko jedynki.

Z tego właśnie wykresu, możemy zauważyć, że ${}w{}$ wcale nie wpływa na wynik naszego modelu znacząco. To dlatego, że gdy ten parametr zbliża się do jedynki to maleje jego znaczenie. Classical overhead zamienia się wtedy w zwykły bias, czyli przesunięcie granicy thresholdu. Z tego mechanizmu korzysta model przy datasecie circles. Zatem mechanizm treningowy, na którym opierała się hipoteza, nie zaszedł. Zaobserwowany efekt to zamiast tego mechanizm kalibracji progu.

Dla danych moons różnica między wariantami jest jeszcze mniej wyraźna — zarówno accuracy, jak i mean absolute f(theta) są niemal identyczne dla obu modeli, co sugeruje, że efekt zaobserwowany dla circles nie jest uniwersalny i może zależeć od konkretnego kształtu/trudności problemu.

Na podstawie analizy wykresu accuracy dla Circles widać, że zastosowania overhead miało efekt uboczny. Mimo, że finalnie wartości są podobne, to początkowe accuracy, na którym model overhead się niemalże utrzymuje, jest znacznie wyższe niż accuracy modelu threshold.

Podsumowując powyższe spostrzeżenia wnioskuję, że doświadczenie nie obaliło, ani nie potwierdziło hipotezy. Jako dalsze kroki warto rozważyć np. rozdzielenie learning rate dla ${}\theta{}$ oraz ${}w{}$ i ${}b{}$, by wykluczyć wpływ niedopasowanej szybkości uczenia obu grup parametrów.

Bruno Niklasiewicz
