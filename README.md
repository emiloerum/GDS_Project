# GDS Project - Fake News Detecter
Dette github repository indeholder kildekoden af vores udarbejdelse af en fake news detecter til eksamensprojektet i kurset "Grundlæggende Data Science".
Repositoriet fungerer som et opslagsværk som supplement til vores projekt rapport - der omhandler processesen af vores arbejde og præsenterer resultaterne heraf - snarere end et eksekverbart projekt. Projektet er lavet i jupyter notebooks og en enkelt python fil.
Dette skyldes at det ikke er muligt at lægge alt den anvendte data - der i kildekoden bliver preprocesseret og modelleret - op i repositoriet da filerne er meget store.

Nedenstående struktur giver kun et kort resume af indholdet af filerne.
## Struktur
- Exploration_and_Preprocessesing/
  - boxplot.ipynb/': Kode til generering af boxplots over artikellængder for falske og sande nyheder
  - exploration.ipynb/': Udforskning af data
  - largedatapreprocessing.ipynb/': Cleaning og preprocessering af datasættet "995,000_rows.csv"
  - projecttask1.ipynb/': Preprocessering af subset af "995,000_rows.csv", diverse mellemresultater og udforskning af dette.
  - scraped_data_preprocessing.ipynb:/': Preprocessering af data scraped fra BBC i "Individual Graded Exercise 2"
- Models_and_Evaluation/
  - baseline1_lenofarticle.ipynb/': Logistisk regression med længde feature og præstation af denne på validation, test samt LIAR-test sæt. (inklusiv baseline trænet på extended data)
  - baseline2_factcount.ipynb/': Logistisk regression med "fact-count" feature og evaluering af denne på validation, test samt LIAR-test sæt. (inklusiv baseline trænet på extended data)
  - trial_adv.ipynb/': MCP-classifiers trænet på data med tf-idf repræsentation og evaluering af disse på validation, test samt LIAR-test sæt.
- data/': De fleste filer i denne mappe er ignoreret af .gitignore da de er for store at have i repositoriet
  - LIAR/': mappe indeholdende det originale LIAR datasæt i et train, validation, test split samt README
  - labeled_liar_statements_preprocessed.pkl/': pickle fil indeholdende liar-data preprocesseret
- LIAR_preprocessing.ipynb/': Preprocessering af LIAR datasættet.
- extended_content_creation.ipynb/': Kode til generering af preprocesseret corpus forlænget med preprocesseret scraped data fra BBC.
- functions.py/': Python modul indeholdende hjælpefunktioner defineret til data preprocessering og modellering
