Zadanie rekrutacyjne – System do wykrywania kształtów

Kandydat przygotowuje aplikacje z prostym interfejsem graficznym (np. w Streamlit, PyQt, Gradio – wybór należy do Ciebie), która umożliwia:

- Wczytanie obrazu dokumentu (skan/zdjęcie).

- Automatyczne wykrywanie wszystkich kształtów na dokumencie z użyciem modelu uczenia maszynowego (np. gotowego detektora obiektów, własnego lekkiego modelu CNN albo innej architektury). Interesują nas trzy klasy figur: kwadrat, koło, trójkąt.

- Obsługę niepewnych przypadków – jeśli model wykryje figurę z niską pewnością lub błędną klasyfikacją, użytkownik może ręcznie wskazać poprawną klasę w GUI.

- Inkrementalne douczanie modelu – ręcznie poprawione przypadki trafiają do zbioru i model powinien potrafić je wykorzystać do dalszego douczania bez pełnego trenowania od zera.

- Prezentację wyników – podgląd wykrytych kształtów na obrazie, opcjonalne sprawdzenie poprawności (np. parzystość lub krótki checksum).


Zadanie dodatkowe (dla chętnych)

Rozszerz swoją aplikację tak, aby umożliwiała wykrywanie i dekodowanie kształtów w czasie rzeczywistym z kamery:

Aplikacja powinna umożliwiać włączenie podglądu z kamery (np. cv2.VideoCapture albo moduły webowe w Streamlit/PyQt).

Na każdym klatce wideo należy wykonywać detekcję kształtów tym samym modelem, którego używasz dla obrazów statycznych.

Użytkownik powinien widzieć na żywo podgląd obrazu z naniesionymi detekcjami (bounding boxy, etykiety, pewności).


Oczekiwane elementy rozwiązania

- Repozytorium z kodem w Pythonie, zawierające:

- logikę modelu (wykorzystanie istniejącej architektury lub własnej),

- GUI z możliwością wgrywania obrazów i adnotacji,

- mechanizm zapisu danych treningowych i douczania modelu.

- Instrukcję uruchomienia (README) oraz przykładowy pipeline: jak przygotować dane, uruchomić aplikację, douczyć model.

Testowe dane wejściowe dostarczone lub możesz samodzielnie wygenerować syntetyczny zestaw figur (np. kwadraty, koła i trójkąty w różnych wariantach, rotacjach, z szumem).


Kryteria oceny

- Jakość i stabilność działania aplikacji (czy wykrywa i klasyfikuje figury).

- Poprawne działanie mechanizmu ręcznej adnotacji i douczania.

- Architektura kodu – czytelność, modularność, testowalność.

- Świadomy dobór rozwiązań ML (np. dlaczego wybrany został dany model i jak radzi sobie z douczaniem).

Dane dostarczone:
- plik z danymi przed obróbką "ShapeDetector raw"
- plik z danymi po obróbce obrazu "ShapeDetector treshold"