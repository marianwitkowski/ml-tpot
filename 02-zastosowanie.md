## 2. Zastosowanie TPOT w praktyce

### 2.1. Instalacja i konfiguracja

Aby rozpocząć pracę z TPOT, najpierw musimy zainstalować narzędzie oraz upewnić się, że nasza konfiguracja środowiska jest poprawna. TPOT jest narzędziem napisanym w Pythonie, które bazuje na bibliotece scikit-learn. Oznacza to, że wymaga ona zainstalowanych zależności, takich jak numpy, scipy oraz pandas. TPOT jest kompatybilny z Pythonem w wersji 3.6 lub wyższej.

#### Kroki instalacji TPOT:

1. **Instalacja TPOT**:

   Aby zainstalować TPOT, wystarczy użyć polecenia `pip`:

   ```bash
   pip install tpot
   ```

   Możemy także dodać inne biblioteki wspierające TPOT, takie jak `xgboost`, jeśli chcemy używać dodatkowych algorytmów w procesie automatycznej optymalizacji pipeline.

2. **Sprawdzenie instalacji**:

   Po zakończonej instalacji warto upewnić się, że TPOT został zainstalowany poprawnie. Możemy to zrobić, importując bibliotekę w sesji Pythona:

   ```python
   from tpot import TPOTClassifier
   ```

   Jeśli import zakończy się sukcesem, oznacza to, że instalacja przebiegła pomyślnie.

3. **Zalecana konfiguracja środowiska**:

   TPOT może wymagać sporych zasobów obliczeniowych, zwłaszcza jeśli ustawimy dużą populację lub liczbę generacji. Dlatego zalecane jest uruchamianie narzędzia na maszynach z dużą ilością pamięci RAM oraz procesorem wielordzeniowym. Można także skonfigurować użycie GPU dla algorytmów wspierających takie akceleratory, takich jak `XGBoost`.

### 2.2. Przykład użycia na rzeczywistym zbiorze danych

Teraz omówimy, jak zastosować TPOT w praktyce, korzystając z rzeczywistego zbioru danych. Dla naszego przykładu wykorzystamy popularny zbiór danych "Titanic", który zawiera informacje o pasażerach statku, w tym czy przeżyli katastrofę. Naszym celem będzie zbudowanie modelu klasyfikacji, który przewiduje, czy pasażer przeżył.

#### Kroki:

1. **Ładowanie zbioru danych**:

   Pobieramy zbiór danych Titanic, który dostępny jest w wielu źródłach (np. w bibliotece `seaborn` lub Kaggle). Na potrzeby tego przykładu, załadujemy dane z pliku CSV.

2. **Preprocessing**:

   Aby TPOT działał poprawnie, musimy upewnić się, że wszystkie dane są w odpowiednim formacie. Na przykład, musimy przekonwertować zmienne kategoryczne na wartości liczbowe oraz usunąć brakujące dane.

3. **Uruchomienie TPOT**:

   Teraz uruchomimy TPOT, aby zoptymalizować pipeline oraz wybrać najlepszy model.

#### Przykład kodu:

```python
# Instalacja bibliotek
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Załadowanie danych Titanic z pliku CSV
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# Preprocessing danych
titanic = titanic.drop(['Name', 'Ticket', 'Cabin'], axis=1)  # Usunięcie zbędnych kolumn
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})  # Konwersja płci na wartości liczbowe
titanic['Embarked'].fillna('S', inplace=True)
titanic['Embarked'] = titanic['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # Konwersja portu
titanic['Age'].fillna(titanic['Age'].median(), inplace=True)  # Uzupełnienie braków w wieku
titanic['Fare'].fillna(titanic['Fare'].median(), inplace=True)  # Uzupełnienie braków w cenie biletu

# Definicja zmiennych X i y
X = titanic.drop('Survived', axis=1)  # Wszystkie cechy oprócz kolumny 'Survived'
y = titanic['Survived']  # Zmienna docelowa

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicjalizacja TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

# Trening modelu TPOT
tpot.fit(X_train, y_train)

# Ocena modelu na zbiorze testowym
print(f'Dokładność na zbiorze testowym: {tpot.score(X_test, y_test)}')

# Eksport najlepszego pipeline do pliku
tpot.export('tpot_titanic_pipeline.py')
```

#### Analiza kodu:

1. **Ładowanie i czyszczenie danych**: Na początku wczytujemy zbiór danych Titanic, usuwamy niepotrzebne kolumny, przekształcamy zmienne kategoryczne (np. płeć i miejsce zaokrętowania) na wartości numeryczne oraz uzupełniamy brakujące wartości w kolumnach "Age" i "Fare". Dzięki temu nasz zbiór danych jest gotowy do użycia przez algorytmy uczenia maszynowego.

2. **Podział danych**: Dzielimy dane na zbiór treningowy i testowy przy użyciu funkcji `train_test_split` z biblioteki scikit-learn. Używamy 80% danych do trenowania modelu i 20% do jego oceny.

3. **Trening TPOT**: Używamy klasy `TPOTClassifier` z 5 generacjami i populacją 20 pipelines w każdej generacji. Oznacza to, że TPOT przetestuje 100 różnych pipelines (20 na generację przez 5 generacji).

4. **Ocena modelu**: Po treningu TPOT ocenia najlepszy pipeline na zbiorze testowym, podając dokładność modelu.

5. **Eksport modelu**: TPOT automatycznie eksportuje kod Python dla najlepszego znalezionego pipeline do pliku `tpot_titanic_pipeline.py`, co umożliwia jego późniejsze wykorzystanie.

---

(c) 2024 Marian Witkowski - wszelkie prawa zastrzeżone
