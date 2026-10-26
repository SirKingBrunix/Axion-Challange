#### Problem badawczy
Czy sieć korzystająca jedynie z funkcji aktywacji ReLU w warstwach ukrytych, będzie miała wyższe wyniki (zdefiniowane jako dokładność na zbiorze testowym) niż ta korzystająca jedynie z funkcji aktywacji sigmoida w warstwach ukrytych? Czy najoptymalniejsze hiperparametry (z pewnego wybranego zakresu przeszukiwania) będą takie same dla obu funkcji? Jak z zadanym problemem poradzi sobie algorytm kNN?
#### Hipoteza
- Sieć neuronowa z funkcją ReLU będzie miała wyższe wyniki niż sieć z sigmoidą
- Najlepsze hiperparametry będą różne dla obu funkcji
- Algorytm kNN będzie miał mniejszą dokładność niż obie sieci neuronowe
***
#### Plan doświadczenia:
1. Wybrać dataset z klasyfikacją binarną
2. Stworzyć model kNN i zoptymalizować jego wyniki
3. Stworzyć sieć neuronową z funkcjami aktywacji ReLU i znaleźć najlepszą konfigurację
4. Stworzyć sieć neuronową z funkcjami aktywacji sigmoida i znaleźć najlepszą konfigurację
5. Porównać wyniki i konfiguracje
6. Sformułować wnioski i obserwacje
***
#### Dataset
Doświadczenie będzie wykonywane na zbiorze danych z próbkami pacjentów na podstawie, którego model będzie starał się określić czy osoba ma chorobę serca czy też jej nie ma. Opis danych ze źródła (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset):
	This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.
Baza zawiera 1025 próbek.
#### Exploratory Data Analysis
Przykładowe histogramy cech:


<img src="Images/Pasted%20image%2020261025154701.png" alt="Pasted image 20261025154701.png" width="335">
<img src="Images/Pasted%20image%2020261025154723.png" alt="Pasted image 20261025154723.png" width="335">
<img src="Images/Pasted%20image%2020261025154738.png" alt="Pasted image 20261025154738.png" width="335">
<img src="Images/Pasted%20image%2020261025154853.png" alt="Pasted image 20261025154853.png" width="335">

Na podstawie przeprowadzonej analizy eksploracyjnej można zauważyć, że niektóre cechy wykazują różnice w rozkładach pomiędzy osobami z chorobą a osobami bez rozpoznanej choroby. Przykładem są `thalach`, `oldpeak` oraz `thal`. Jednocześnie rozkłady części cech nakładają się na siebie, dlatego na tym etapie nie można jednoznacznie określić, jak skutecznie model będzie rozróżniał obie klasy.

Zaobserwowane różnice stanowią podstawę do stworzenia modeli, które z pewnym prawdopodobieństwem będą szacowały przynależność do danej klasy.

Powyższe dane skalujemy i dzielimy na trzy rozłączne zbiory:
- train - do trenowania modeli, czyli dopasowania ich parametrów do danych
- validation - wykorzystywany do oceny i porównywania modeli
- test - wykorzystywany do końcowej oceny wybranego modelu na danych, które nie brały udziału w jego trenowaniu ani doborze hiperparametrów
Następnie przeprowadzamy skalowanie cech numerycznych za pomocą funkcji StandardScaler z biblioteki sklearn.

#### Analiza balansu klas
Sprawdzamy liczebność poszczególnych klas zmiennej docelowej. Pozwala to ocenić, czy zbiór danych jest zbalansowany, czy też jedna z klas występuje znacznie częściej od drugiej. Jeżeli proporcje klas są zbliżone, nie ma początkowej potrzeby stosowania oversamplingu.

<img src="Images/Pasted%20image%2020261026154757.png" alt="Pasted image 20261026154757.png" width="500">

Jako, że wyniki są zbliżone, to decydujemy się nie korzystać z metody oversamplingu.
***
#### k-Nearest Neighbors (kNN)
Algorytm patrzy na k najbliższych punktów w przestrzeni cech i na podstawie tego jak sklasyfikowana jest większość z tych k sąsiadów definiuje predykcję aktualnie sprawdzanego punktu.

Korzystam z zaprogramowanego modelu z biblioteki sklearn, dla którego ustawiam parametry. Na początek n_neighbours = 1:

**Wyniki:**



<img src="Images/Pasted%20image%2020261025114155.png" alt="Pasted image 20261025114155.png" width="500">
***
Obserwacja:
wyniki algorytmu kNN dla (n_neighbours = 1) są zaskakująco wysokie. Może być to spowodowane zjawiskiem Data Leakage. Jeżeli w zbiorze testowym są powtórki ze zbioru treningowego, to model znajduje punkty niemal idealnie:





<img src="Images/Pasted%20image%2020261025122003.png" alt="Pasted image 20261025122003.png" width="500">

Zatem trzeba pozbyć się duplikatów ze zbioru danych za pomocą komendy:  
	df = df.drop_duplicates().reset_index(drop=True)
Niestety z tego powodu liczba rekordów zmniejszyła się do 302. Ponownie stosujemy analizę balansu klas z wynikami poniżej:




<img src="Images/Pasted%20image%2020261026155155.png" alt="Pasted image 20261026155155.png" width="500">

Na ich podstawie decyduję się ponownie na na nie korzystanie z metody oversampling.
***
Sprawdzamy wyniki dla $n_{\text{neighbors}} \in \{1, 3, 5\}$. Otrzymujemy teraz realniejsze wyniki. Po sprawdzeniu zarówno zbiorze walidacyjnym najlepsze dla $n_{\text{neighbors}} = 5$:

Testowe wyniki:




<img src="Images/Pasted%20image%2020261025134832.png" alt="Pasted image 20261025134832.png" width="500">



Walidacyjne wyniki:




<img src="Images/Pasted%20image%2020261025134847.png" alt="Pasted image 20261025134847.png" width="500">

***
#### Sieć Neuronowa z ReLU

Sieć neuronowa - algorytm stworzony z warstw neuronowych, które przetwarzają dane. Warstwa pierwsza przyjmuje dane. Następne (ukryte) warstwy przetwarzają jakie wnioski płyną z warstwy pierwszej. Ostatnia warstwa zwraca predykcję. Do implementacji sieci wykorzystuję bibliotekę TensorFlow, która umożliwia definiowanie architektury sieci, dobór parametrów oraz przeprowadzenie procesu trenowania.
$\text{ReLU}(x)=\max(0,x)$

Pierwszy prototyp struktury wygląda następująco z funkcją aktywacji ReLU (Rectified Linear Unit):
$$\text{ReLU}(x)=\max(0,x)$$




<img src="Images/Pasted%20image%2020261025185249.png" alt="Pasted image 20261025185249.png" width="500">

Jako, że nasz wynik to binarna klasyfikacja, to w warstwie wyjściowej wystarczy użyć jednego neuronu z funkcją aktywacji sigmoid, by otrzymać wynik. Do uzyskania końcowej klasy stosuję próg decyzyjny równy 0,5. Jeśli przewidywane prawdopodobieństwo jest mniejsze niż 0,5, model przypisuje klasę 0. W przeciwnym przypadku przypisuje klasę 1.

Użytą funkcją straty będzie binary_crossentropy (BCE), przeznaczona do klasyfikacji binarnej:
$$
BCE = - \frac{1}{N} \sum_{i=1}^{N}[y_i ln(p_i) + (1 - y_i)ln(1-p_i)]
$$
- $N$ jest liczbą próbek
- $y_i$ jest prawdziwą klasą danej próbki (0 lub 1)
- $p_i$ jest przewidywanym prawdopodobieństwem $i$-tej próbki bycia w klasie 1
Jest to funkcja straty używana w modelach, które klasyfikują binarnie.

Korzystam również z funkcji EarlyStopping, która zatrzymuje proces uczenia modelu w momencie, gdy funkcja straty dla splitu walidacyjnego (części zbioru treningowego, który wyznaczamy w celu określania val_loss) nie zmniejsza się przez zadaną liczbę epok. Dla powyższego modelu i parametrów:
- epochs = 50
- batch_size = 8
Otrzymujemy wyniki (przy czym val_loss jest liczony na podstawie splitu danych testowych, więc może się znacząco różnić od wartości funkcji straty dla danych walidacyjnych):


<img src="Images/Pasted%20image%2020261025195044.png" alt="Pasted image 20261025195044.png" width="500">

Poniższe dane są liczone na testowej części danych:



<img src="Images/Pasted%20image%2020261025195103.png" alt="Pasted image 20261025195103.png" width="500">

W kolejnym eksperymencie rozszerzam architekturę o funkcję dropout by wyłączać z zadanym prawdopodobieństwem pewne neurony w trakcie uczenia. To ogranicza nadmierne uzależnienie modelu od konkretnych jednostek sieci.

Celem tej modyfikacji jest ograniczenie overfittingu, czyli sytuacji, w której model zbyt dobrze dopasowuje się do danych treningowych, ale gorzej generalizuje na nowych obserwacjach.
***
#### Grid search:
By dopasować najlepsze parametry stosujemy metodę grid search dla poniższych wartości:
- Liczba neuronów w każdej warstwie $\in\{ 8, 16, 32\}$
- Prawdopodobieństwo pominięcia neuronu $\in\{ 0, 0.1, 0.3 \}$
- Learning rate $\in\{ 0.001, 0.005 \}$
- Batch size $\in\{ 8, 16, 32 \}$
Staram się znaleźć modele za pomocą dwóch kryteriów. Jeden model o najmniejszej wartości straty dla zbioru danych walidacyjnych. Drugi model o największej dokładności dla zbioru danych walidacyjnych.

#### ReLU:
- Model minimalnej straty:
	- parametry:
		- 16 - liczba neuronów
		- 0.3 - prawdopodobieństwo pominięcia neuronu
		- 0.005 - learning rate
		- 8 - batch size
	- wartość funkcji straty dla danych walidacyjnych: ${}{}$ ${\approx}{0.51}$
	- wykresy treningowe:



	  <img src="Images/Pasted%20image%2020261026142251.png" alt="Pasted image 20261026142251.png" width="500">
	- wyniki na danych testowych:



	 	 <img src="Images/Pasted%20image%2020261026142323.png" alt="Pasted image 20261026142323.png" width="639">
- Model maksymalnej dokładności:
	- parametry:
		- 16 - liczba neuronów
		- 0.3 - prawdopodobieństwo pominięcia neuronu
		- 0.005 - learning rate
		- 32 - batch size
	- wartość funkcji straty dla danych walidacyjnych: ${}{}$ ${\approx}{0.83}$
	- wykresy treningowe:

 - 
	  	  <img src="Images/Pasted%20image%2020261026143113.png" alt="Pasted image 20261026143113.png" width="500">
	- wyniki na danych testowych:


	 <img src="Images/Pasted%20image%2020261026143129.png" alt="Pasted image 20261026143129.png" width="643">

#### Sigmoid:
- Model minimalnej straty:
	- parametry:
		- 8 - liczba neuronów
		- 0.3 - prawdopodobieństwo pominięcia neuronu
		- 0.005 - learning rate
		- 16 - batch size
	- wartość funkcji straty dla danych walidacyjnych: ${}{}$ ${\approx}{0.51}$
	- wykresy treningowe:


	  <img src="Images/Pasted%20image%2020261026143627.png" alt="Pasted image 20261026143627.png" width="500">
	- wyniki na danych testowych:


	  <img src="Images/Pasted%20image%2020261026143648.png" alt="Pasted image 20261026143648.png" width="500">
- Model maksymalnej dokładności:
	- parametry:
		- 8 - liczba neuronów
		- 0 - prawdopodobieństwo pominięcia neuronu
		- 0.001 - learning rate
		- 32 - batch size
	- wartość funkcji straty dla danych walidacyjnych: ${}{}$ ${\approx}{0.78}$
	- wykresy treningowe:


	  <img src="Images/Pasted%20image%2020261026143752.png" alt="Pasted image 20261026143752.png" width="500">
	- wyniki na danych testowych:


	 <img src="Images/Pasted%20image%2020261026143803.png" alt="Pasted image 20261026143803.png" width="500">
***
### Wnioski
Zestawię wszystkie dokładności omawianych modeli:
- Model kNN - 79%
- Przykładowy model sieci neuronowej - 82%
- Model ReLU minimalnej straty - 77%
- Model ReLU maksymalnej dokładności - 77%
- Model Sigmoid minimalnej straty - 84%
- Model Sigmoid maksymalnej dokładności - 80%

To zestawienia pozwala dostrzec pewną powtarzalność. Okazuje się, że właściwie wszystkie sposoby uczenia maszynowego mają bardzo podobne wyniki. Żaden nie jest wyraźnie lepszy.

Na pierwszy rzut oka Model Sigmoid minimalnej straty ma najwyższą dokładność, jednak trzeba zwrócić uwagę na dane, na których opierało się doświadczenie. Po zmniejszeniu bazy danych do jedynie 300 rekordów i jej podzieleniu, w celu ekstrakcji zbiorów do treningu, walidacji i testów, ilość próbek staje się bardzo mała.

Warto zauważyć, że model o przypadkowo dobranych danych różni się jedynie odrobinę od "najlepszego" modelu na liście. Głównie wpływa na to długość listy z próbkami. Żaden model nie ma poniżej 70%, a zatem klasyfikacja przebiega do pewnej dokładności poprawnie (biorąc pod uwagę jedynie kryterium dokładności). Modele o wyższych wynikach właściwie przez przypadek wstrzelają się w zbiór danych testowych. Gdybyśmy stworzyli nowe modele o tych samych parametrach, ich wyniki mogłyby się różnić, od tych których są zaprezentowane powyżej.

Ponadto dane obrazują fakt, że wyższość sieci neuronowych nie jest wcale zawsze oczywista. Przy takich danych i omawianym problemie, model kNN radzi sobie porównywalnie dobrze co sieci neuronowe. W dodatku można wyciągnąć wniosek, że w ramach przetestowanych konfiguracji nie udało się uzyskać wyraźnie lepszych wyników.

Nie stwierdzam jednak, że różnice w dokładnościach wynikają jedynie z małej ilości danych do trenowania. Sądzę jednak, że na wiele pytań zadanych w problemie badawczym nie da się jednoznacznie odpowiedzieć przy tym zbiorze danych.
***
### Skontrowanie hipotezy
Hipoteza (dla przypomnienia):
- Sieć neuronowa z funkcją ReLU będzie miała wyższe wyniki niż sieć z sigmoidą
- Najlepsze hiperparametry będą różne dla obu funkcji
- Algorytm kNN będzie miał mniejszą dokładność niż obie sieci neuronowe
Doświadczenie:
1. Sieci neuronowe mają zbliżone wyniki. Dla tak małego zbioru danych nie jestem w stanie stwierdzić jak zmiana funkcji aktywacji wpływ na wynik, który jest w dużym stopniu zależny od przypasowania się modelu do danych testowych.
2. Parametry okazały się różne. Trzeba zaznaczyć, że dla tych samych parametrów, ale różnych funkcji aktywacji, modele prawdopodobnie miałyby zbliżone wartości dokładności do modeli, które zostały przetestowane. Ponownie, ze względu na mały rozmiar bazy danych nie ma możliwości klarownego udowodnienia tego punktu, więc analiza na większym zbiorze mogłaby wykazać duże różnice hiperparametryczne lub pokazać, że są one niewielkie.
3. Model kNN wcale nie okazał się być słabszy od sieci neuronowych. To pokazuje, że do pewnych prostych klasyfikacji, szczególnie na niewielkich zbiorach danych, modele nie będące sieciami neuronowymi mogą sprawdzać się równie dobrze, a są mają znacząco prostszą strukturze.

															Wrocław
															Bruno Niklasiewcz
