1.Żeby uruchomić kod należy zainstalować biblioteki dostępne w requirements.txt komendą
pip install -r ./requirements.txt
2. Po instalacji wszystkich zależności należy w terminalu powershell wywołać komendę 'streamlit run main.py'
Jeżeli program uruchomił się poprawnie to powinien otworzyć przeglądarkę internetową na adresie http://localhost:8502/

Dane można wgrać w postaci dowolnego obrazu z kształtami do detekcji.
Po wgraniu i analizie obrazu program zapyta czy użytkownik chce poprawić etykiety na tych nie spełniających progu pewności programu, a po edycji wszystkich zapyta o dotrenowanie modelu.

Na ten moment jestem świadom wielu słabości programu.
Nie rozpoznaje wszystkich kształtów do klasyfikacji. - Muszę się przyjrzeć modelowi detekcji i go dopracować.
Nie klasyfikuje prawidłowo większości kształtów. - Podobnie do powyższego.
Po pytaniu ręczną klasyfikację zdaje się powtarzać oznaczone już kształty i po przeklikaniu wszystkich (chyba) ostatni zostawia na ekranie mając jednocześnie pytanie o douczanie modelu.
Metryki walidacyjne (acc, prec, recall) są znacząco poniżej akcpetowalnych - liczę że po poprawieniu pierwszych dwóch punktów tutaj też będzie lepiej.

Dane treningowe ściągnąłem z kaggle'a i na tym działałem: https://www.kaggle.com/datasets/vijay20213/shape-detection-circle-square-triangle/data