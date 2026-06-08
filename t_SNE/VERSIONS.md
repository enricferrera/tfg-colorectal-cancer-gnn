# Historial de Versions t-SNE

Aquest fitxer detalla els canvis realitzats en cada versió de l'eina de visualització t-SNE situada a `enric/t_SNE/`.

### v1: Implementació Inicial
- Creació del mòdul `t_SNE.py`.
- Generació d'un gràfic t-SNE global amb tots els patches del dataset.
- Paràmetres estàndard de `scikit-learn` (CPU, single-core).

### v2: Optimització i Millora Visual
- **Rendiment**: Activació de `n_jobs=-1` (multicore) i inicialització `pca`.
- **Estètica**: Reducció de la mida del punt (`s=2`) i augment de transparència (`alpha=0.3`) per evitar el "blob" d'informació.
- **Algorisme**: Augment de la perplexitat a `50` per al gràfic global per capturar millor l'estructura.
- **Organització**: Creació de la funció `main` per generar gràfics individuals per pacient.

### v3: Detecció d'Escala i Mostreig Equilibrat
- **Lògica**: Detecció automàtica de l'escala de la variable `affectation` (0-1 o 0-100) per evitar que tot surti verd per error de llindar.
- **Subsampling**: Implementació de "Balanced Subsampling" per al global, prioritzant patches infiltrats i de frontera perquè apareguin colors vermells i taronges.

### v4: Retorn al Mostreig Aleatori
- **Lògica**: Retorn al mostreig aleatori pur per al global (30.000 patches) per comprovar la distribució real un cop corregida l'escala dels llindars.
- Manteniment de la detecció d'escala intel·ligent.

### v5: Mostreig Estratificat i Estructura Jeràrquica (Versió Professional)
- **Global**: Mostreig estratificat (fins a 10.000 punts per cada categoria: Sa, Infiltrat, Front) per garantir una visió completa de totes les estructures.
- **Pacients**: Limitació a 10 pacients representatius per optimitzar temps i anàlisi.
- **Organització**: Nova estructura de carpetes: `results/vX/` per al global i `results/vX/patients/` per als individuals.
- **Reproductibilitat**: Integració total amb `fix_seeds(123)`.

### v6: Sistema de Versionat Automàtic
- Implementació de la funció `get_next_version` per detectar automàticament la següent carpeta `vX` disponible.
- Eliminació de la lògica de "SKIP" per permetre execucions consecutives amb nous paràmetres.

### v7: Selecció Intel·ligent de Pacients (Bugfix N1)
- **Diagnòstic**: S'ha detectat que molts pacients N1 tenen un 0% d'infiltració local en els patches del dataset (Zones sanes de pacients malalts).
- **Lògica**: S'ha implementat un algorisme que ordena els pacients N1 per quantitat de teixit infiltrat real.
- **Objectiu**: Garantir que els gràfics individuals de pacients N1 mostrin realment l'estructura del tumor (punts vermells/taronges) i no només teixit sa adjacent.
- **Consistència**: Ús de llindars globals (0.1 i 0.9) per a tota l'execució.
