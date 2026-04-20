
"""
Feature Engineering - Projet 7 : Classification des reprises anaphoriques erronées
Auteure : Sarah

Nouvelles features créées :
- Genre_reprise          : genre grammatical de la reprise (M/F/N/inconnu)
- Nombre_reprise         : nombre grammatical de la reprise (S/P/inconnu)
- Genre_antecedent       : genre grammatical de l'antécédent (M/F/N/inconnu)
- Nombre_antecedent      : nombre grammatical de l'antécédent (S/P/inconnu)
- Match_genre            : accord en genre reprise/antécédent (1/0/-1 si inconnu)
- Match_nombre           : accord en nombre reprise/antécédent (1/0/-1 si inconnu)
- Longueur_reprise       : nombre de mots dans la reprise
- Longueur_antecedent    : nombre de mots dans l'antécédent
- Est_pronom             : 1 si la reprise est un pronom, 0 sinon
- Type_pronom_detaille   : sous-type du pronom (personnel, démonstratif, relatif, possessif, indéfini, autre)
"""

import pandas as pd
import re

# ---------------------------------------------------------------------------
# 1. Dictionnaires morphologiques
# ---------------------------------------------------------------------------

# Pronoms avec genre et nombre connus
PRONOMS_MORPHO = {
    # Personnels
    "il":        ("M", "S", "personnel"),
    "elle":      ("F", "S", "personnel"),
    "ils":       ("M", "P", "personnel"),
    "elles":     ("F", "P", "personnel"),
    "le":        ("M", "S", "personnel"),
    "la":        ("F", "S", "personnel"),
    "les":       ("N", "P", "personnel"),
    "lui":       ("M", "S", "personnel"),
    "leur":      ("N", "P", "personnel"),
    "y":         ("N", "S", "personnel"),
    "en":        ("N", "S", "personnel"),
    "se":        ("N", "N", "personnel"),
    "me":        ("N", "N", "personnel"),
    "te":        ("N", "N", "personnel"),
    "nous":      ("N", "P", "personnel"),
    "vous":      ("N", "P", "personnel"),

    # Démonstratifs
    "ce":           ("M", "S", "démonstratif"),
    "cet":          ("M", "S", "démonstratif"),
    "cette":        ("F", "S", "démonstratif"),
    "ces":          ("N", "P", "démonstratif"),
    "celui":        ("M", "S", "démonstratif"),
    "celle":        ("F", "S", "démonstratif"),
    "ceux":         ("M", "P", "démonstratif"),
    "celles":       ("F", "P", "démonstratif"),
    "celui-ci":     ("M", "S", "démonstratif"),
    "celui-là":     ("M", "S", "démonstratif"),
    "celle-ci":     ("F", "S", "démonstratif"),
    "celle-là":     ("F", "S", "démonstratif"),
    "ceux-ci":      ("M", "P", "démonstratif"),
    "ceux-là":      ("M", "P", "démonstratif"),
    "celles-ci":    ("F", "P", "démonstratif"),
    "celles-là":    ("F", "P", "démonstratif"),
    "ceci":         ("N", "S", "démonstratif"),
    "cela":         ("N", "S", "démonstratif"),
    "ça":           ("N", "S", "démonstratif"),

    # Relatifs
    "qui":      ("N", "N", "relatif"),
    "que":      ("N", "N", "relatif"),
    "qu":       ("N", "N", "relatif"),
    "dont":     ("N", "N", "relatif"),
    "où":       ("N", "N", "relatif"),
    "auquel":   ("M", "S", "relatif"),
    "auxquels": ("M", "P", "relatif"),
    "auxquelles":("F", "P", "relatif"),
    "laquelle": ("F", "S", "relatif"),
    "lequel":   ("M", "S", "relatif"),
    "lesquels": ("M", "P", "relatif"),
    "lesquelles":("F", "P", "relatif"),
    "duquel":   ("M", "S", "relatif"),

    # Possessifs
    "son":      ("M", "S", "possessif"),
    "sa":       ("F", "S", "possessif"),
    "ses":      ("N", "P", "possessif"),
    "leur":     ("N", "S", "possessif"),
    "leurs":    ("N", "P", "possessif"),
    "mon":      ("M", "S", "possessif"),
    "ma":       ("F", "S", "possessif"),
    "mes":      ("N", "P", "possessif"),
    "ton":      ("M", "S", "possessif"),
    "ta":       ("F", "S", "possessif"),
    "tes":      ("N", "P", "possessif"),
    "notre":    ("N", "S", "possessif"),
    "votre":    ("N", "S", "possessif"),
    "nos":      ("N", "P", "possessif"),
    "vos":      ("N", "P", "possessif"),
    "le mien":  ("M", "S", "possessif"),
    "la mienne":("F", "S", "possessif"),
    "le sien":  ("M", "S", "possessif"),
    "la sienne":("F", "S", "possessif"),

    # Indéfinis fréquents
    "l'un":     ("M", "S", "indéfini"),
    "l'une":    ("F", "S", "indéfini"),
    "l'autre":  ("N", "S", "indéfini"),
    "les deux": ("N", "P", "indéfini"),
    "certain":  ("M", "S", "indéfini"),
    "certains": ("M", "P", "indéfini"),
    "certaine": ("F", "S", "indéfini"),
    "certaines":("F", "P", "indéfini"),
    "chacun":   ("M", "S", "indéfini"),
    "chacune":  ("F", "S", "indéfini"),
    "aucun":    ("M", "S", "indéfini"),
    "aucune":   ("F", "S", "indéfini"),
    "plusieurs":("N", "P", "indéfini"),
    "tout":     ("M", "S", "indéfini"),
    "tous":     ("M", "P", "indéfini"),
    "toute":    ("F", "S", "indéfini"),
    "toutes":   ("F", "P", "indéfini"),
    "nul":      ("M", "S", "indéfini"),
    "nulle":    ("F", "S", "indéfini"),
}

# Expressions à plusieurs mots connues
EXPRESSIONS_MORPHO = {
    "ce dernier":       ("M", "S", "démonstratif"),
    "cette dernière":   ("F", "S", "démonstratif"),
    "ces derniers":     ("M", "P", "démonstratif"),
    "ces dernières":    ("F", "P", "démonstratif"),
    "ce premier":       ("M", "S", "démonstratif"),
    "cette première":   ("F", "S", "démonstratif"),
    "ces deux derniers":("M", "P", "démonstratif"),
    "l'un":             ("M", "S", "indéfini"),
    "l'une":            ("F", "S", "indéfini"),
    "l'autre":          ("N", "S", "indéfini"),
    "les deux":         ("N", "P", "indéfini"),
    "les uns":          ("M", "P", "indéfini"),
    "les unes":         ("F", "P", "indéfini"),
    "les autres":       ("N", "P", "indéfini"),
}

# Déterminants articles pour inférer le genre/nombre d'un GN
DETERMINANTS = {
    "le":  ("M", "S"), "la": ("F", "S"), "les": ("N", "P"),
    "un":  ("M", "S"), "une": ("F", "S"), "des": ("N", "P"),
    "du":  ("M", "S"), "de la": ("F", "S"), "de l'": ("N", "S"),
    "ce":  ("M", "S"), "cet": ("M", "S"), "cette": ("F", "S"), "ces": ("N", "P"),
    "son": ("M", "S"), "sa": ("F", "S"), "ses": ("N", "P"),
    "mon": ("M", "S"), "ma": ("F", "S"), "mes": ("N", "P"),
    "ton": ("M", "S"), "ta": ("F", "S"), "tes": ("N", "P"),
    "notre": ("N", "S"), "votre": ("N", "S"), "nos": ("N", "P"), "vos": ("N", "P"),
    "leur": ("N", "S"), "leurs": ("N", "P"),
    "au":  ("M", "S"), "aux": ("N", "P"),
}

# ---------------------------------------------------------------------------
# 2. Fonctions d'extraction morphologique
# ---------------------------------------------------------------------------

def normaliser(texte):
    """Normalise le texte pour la recherche dans les dictionnaires."""
    if not isinstance(texte, str):
        return ""
    # Normalise les apostrophes et met en minuscule
    texte = texte.strip().lower()
    texte = texte.replace("'", "'").replace("'", "'")
    return texte


def get_morpho_reprise(texte):
    """
    Retourne (genre, nombre, type_pronom) pour une reprise.
    Cherche d'abord dans les expressions multi-mots, puis mot à mot.
    """
    t = normaliser(texte)
    if not t:
        return ("inconnu", "inconnu", "inconnu")

    # Expressions multi-mots en priorité
    if t in EXPRESSIONS_MORPHO:
        return EXPRESSIONS_MORPHO[t]

    # Pronom simple
    if t in PRONOMS_MORPHO:
        return PRONOMS_MORPHO[t]

    # GN : on regarde le premier mot (déterminant)
    premier_mot = t.split()[0]
    if premier_mot in DETERMINANTS:
        genre, nombre = DETERMINANTS[premier_mot]
        return (genre, nombre, "GN")

    # Règles morphologiques sur les suffixes pour les GN sans déterminant reconnu
    genre, nombre = infer_genre_nombre_suffixe(t)
    return (genre, nombre, "autre")


def infer_genre_nombre_suffixe(texte):
    """
    Inférence genre/nombre par suffixes pour les noms communs.
    Retourne (genre, nombre).
    """
    mots = texte.split()
    dernier_mot = mots[-1] if mots else ""

    # Nombre : pluriel si -s ou -x en fin
    if dernier_mot.endswith(("s", "x")) and len(dernier_mot) > 2:
        nombre = "P"
    else:
        nombre = "S"

    # Genre : féminin si suffixe typique
    suffixes_fem = ("tion", "sion", "ité", "ette", "esse", "ance", "ence",
                    "ure", "euse", "rice", "ière", "elle", "ienne", "onne")
    if dernier_mot.endswith(suffixes_fem):
        genre = "F"
    else:
        genre = "inconnu"

    return (genre, nombre)


def get_morpho_antecedent(texte):
    """
    Retourne (genre, nombre) pour un antécédent (souvent un GN).
    On regarde le premier mot pour les déterminants.
    """
    t = normaliser(texte)
    if not t:
        return ("inconnu", "inconnu")

    # Pronom connu ?
    if t in PRONOMS_MORPHO:
        g, n, _ = PRONOMS_MORPHO[t]
        return (g, n)

    # Expression connue ?
    if t in EXPRESSIONS_MORPHO:
        g, n, _ = EXPRESSIONS_MORPHO[t]
        return (g, n)

    # Déterminant en premier mot
    premier_mot = t.split()[0]
    if premier_mot in DETERMINANTS:
        return DETERMINANTS[premier_mot]

    # Sinon on essaie par suffixe
    return infer_genre_nombre_suffixe(t)


def est_pronom(texte):
    """Retourne 1 si la reprise est un pronom connu, 0 sinon."""
    t = normaliser(texte)
    return 1 if (t in PRONOMS_MORPHO or t in EXPRESSIONS_MORPHO) else 0


def match_morpho(val1, val2):
    """
    Compare deux valeurs morphologiques.
    Retourne 1 (accord), 0 (désaccord), -1 (inconnu).
    """
    if val1 == "inconnu" or val2 == "inconnu" or val1 == "N" or val2 == "N":
        return -1
    return 1 if val1 == val2 else 0


# ---------------------------------------------------------------------------
# 3. Pipeline principal
# ---------------------------------------------------------------------------

def enrichir_dataset(chemin_entree, chemin_sortie):
    df = pd.read_excel(chemin_entree)

    # Nettoyage du nom de colonne TypeErreur (contient parfois \xa0)
    df.columns = [c.strip().replace('\xa0', '') for c in df.columns]

    nouvelles_features = []

    for _, row in df.iterrows():
        reprise = row.get("TexteErreur", "")
        antecedent = row.get("TexteAnte", "")

        # Morphologie reprise
        genre_rep, nombre_rep, type_pron = get_morpho_reprise(reprise)

        # Morphologie antécédent
        if pd.isna(antecedent) or antecedent == "":
            genre_ante, nombre_ante = "inconnu", "inconnu"
        else:
            genre_ante, nombre_ante = get_morpho_antecedent(str(antecedent))

        # Accord
        match_genre = match_morpho(genre_rep, genre_ante)
        match_nombre = match_morpho(nombre_rep, nombre_ante)

        # Longueurs
        longueur_rep = len(str(reprise).split()) if isinstance(reprise, str) else 0
        longueur_ante = len(str(antecedent).split()) if isinstance(antecedent, str) and not pd.isna(antecedent) else 0

        nouvelles_features.append({
            "Genre_reprise":        genre_rep,
            "Nombre_reprise":       nombre_rep,
            "Genre_antecedent":     genre_ante,
            "Nombre_antecedent":    nombre_ante,
            "Match_genre":          match_genre,
            "Match_nombre":         match_nombre,
            "Longueur_reprise":     longueur_rep,
            "Longueur_antecedent":  longueur_ante,
            "Est_pronom":           est_pronom(reprise),
            "Type_pronom_detaille": type_pron,
        })

    df_features = pd.DataFrame(nouvelles_features)
    df_enrichi = pd.concat([df, df_features], axis=1)
    df_enrichi.to_excel(chemin_sortie, index=False)
    print(f"Dataset enrichi sauvegardé : {chemin_sortie}")
    print(f"Nouvelles colonnes : {df_features.columns.tolist()}")
    return df_enrichi


# ---------------------------------------------------------------------------
# 4. Exécution + mini rapport
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    ENTREE = "dataset_erreurs_reprises.xlsx"
    SORTIE = "dataset_enrichi.xlsx"

    df = enrichir_dataset(ENTREE, SORTIE)

    # Mini analyse : les nouvelles features aident-elles à distinguer les classes ?
    col_cible = "TypeErreur1"
    print("\n--- Distribution Match_genre par classe ---")
    print(df.groupby([col_cible, "Match_genre"]).size().unstack(fill_value=0))

    print("\n--- Distribution Match_nombre par classe ---")
    print(df.groupby([col_cible, "Match_nombre"]).size().unstack(fill_value=0))

    print("\n--- Distribution Est_pronom par classe ---")
    print(df.groupby([col_cible, "Est_pronom"]).size().unstack(fill_value=0))

    print("\n--- Distribution Type_pronom_detaille par classe ---")
    print(df.groupby([col_cible, "Type_pronom_detaille"]).size().unstack(fill_value=0))


