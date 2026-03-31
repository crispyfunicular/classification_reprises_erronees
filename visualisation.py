"""
visualisation.py - Module dédié à la génération de rapports visuels

Ce fichier isole la logique de construction de l'interface HTML
pour ne pas surcharger les scripts de Machine Learning et respecter
le principe de séparation des préoccupations.
"""

import pandas as pd


def exporter_erreurs_html(
    y_test, y_pred, index_test, fichier_out="analyse_erreurs.html"
):
    """
    Cette fonction se charge de générer l'interface visuelle (Rapport HTML) pour
    l'analyse humaine (qualitative) des fausses prédictions du modèle.
    """

    # Le fichier Excel brut est rechargé afin d'avoir accès au texte complet des phrases
    # (information originelle qui n'existe plus dans la matrice mathématique 'X').
    df_raw = pd.read_excel("dataset_erreurs_reprises.xlsx")

    # Identification exclusive des phrases pour lesquelles le modèle a fourni une mauvaise réponse
    erreurs_mask = y_test != y_pred
    indices_erreurs = index_test[erreurs_mask]

    # Isolement de ces phrases problématiques et association avec le verdict du modèle
    df_erreurs = df_raw.loc[indices_erreurs].copy()
    df_erreurs["Vraie_Classe"] = y_test[erreurs_mask].values
    df_erreurs["Classe_Predite"] = y_pred[erreurs_mask]

    # Construction structurelle de la page Web (HTML/CSS)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Analyse Qualitative des Erreurs</title>
        <style>
            body {{ font-family: 'Segoe UI', system-ui, sans-serif; background-color: #f0f4f8; color: #334155; margin: 0; padding: 30px; line-height: 1.5; }}
            h1 {{ text-align: center; color: #0f172a; margin-bottom: 30px; font-weight: 800; }}
            .summary {{ background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center; font-size: 1.2em; font-weight: 500; }}
            .card {{ background: white; margin-bottom: 25px; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05); border-left: 6px solid #ef4444; transition: transform 0.2s; }}
            .card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }}
            .card.Problème_Grammatical {{ border-left-color: #3b82f6; }}
            .card.Problème_Antecedent {{ border-left-color: #f59e0b; }}
            .card.Problème_Reprise {{ border-left-color: #8b5cf6; }}
            
            .labels {{ display: flex; justify-content: space-between; margin-bottom: 20px; font-weight: 600; font-size: 0.95em; }}
            .label-true {{ background-color: #dcfce7; color: #166534; padding: 6px 12px; border-radius: 6px; }}
            .label-pred {{ background-color: #fee2e2; color: #991b1b; padding: 6px 12px; border-radius: 6px; border: 1px solid #fecaca; }}
            
            .context-box {{ background-color: #f8fafc; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #e2e8f0; font-size: 0.95em; color: #475569; }}
            .context-box strong {{ color: #1e293b; }}
            
            .highlight-reprise {{ background-color: #fef08a; font-weight: 700; color: #854d0e; padding: 2px 6px; border-radius: 4px; border: 1px solid #fde047; }}
            .highlight-antecedent {{ background-color: #bfdbfe; font-weight: 700; color: #1e40af; padding: 2px 6px; border-radius: 4px; border: 1px solid #93c5fd; }}
            
            .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.9em; }}
            .meta-grid div {{ background: #f1f5f9; padding: 10px 15px; border-radius: 6px; color: #475569; }}
            .meta-grid strong {{ color: #334155; display: block; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
        </style>
    </head>
    <body>
        <h1>Analyse Qualitative des Erreurs</h1>
        <div class="summary">
            Le modèle s'est trompé sur <strong>{len(df_erreurs)}</strong> exemples sur un total de <strong>{len(y_test)}</strong> exemples du jeu de Test.
        </div>
    """

    # Génération d'un encadré (une "carte") pour chaque erreur linguistique repérée
    for idx, row in df_erreurs.iterrows():
        reprise = str(row["TexteErreur"]) if pd.notna(row["TexteErreur"]) else ""
        antecedent = str(row["Antecedent"]) if pd.notna(row["Antecedent"]) else ""
        contexte = str(row["Contexte"]).replace("\\n", "<br>")

        # Surlignage visuel de l'antécédent et de la reprise au sein du texte pour faciliter la lecture humaine
        if antecedent and antecedent != "nan" and antecedent in contexte:
            contexte = contexte.replace(
                antecedent, f'<span class="highlight-antecedent">{{antecedent}}</span>'
            )
        if reprise and reprise != "nan" and reprise in contexte:
            contexte = contexte.replace(
                reprise, f'<span class="highlight-reprise">{{reprise}}</span>'
            )

        html_content += f"""
        <div class="card {row['Vraie_Classe']}">
            <div class="labels">
                <span class="label-true">Vraie Classe : {row['Vraie_Classe']}</span>
                <span class="label-pred">Prédiction : {row['Classe_Predite']}</span>
            </div>
            
            <div class="context-box">
                <strong>Extrait (Contexte) :</strong><br><br>
                {contexte}
            </div>
            
            <div class="meta-grid">
                <div><strong>Reprise</strong> <span class="highlight-reprise">{reprise}</span></div>
                <div><strong>Antécédent</strong> <span class="highlight-antecedent">{antecedent if antecedent != "nan" else "Aucun / Absent"}</span></div>
                <div><strong>Type Erreur (Original)</strong> {row['TypeErreur1']}</div>
                <div><strong>Type de Pronom</strong> {row['Type_pronom']}</div>
                <div><strong>GN concurrents</strong> {row['GN_concurrents']}</div>
                <div><strong>Distance en phrases</strong> {row['Distance_phrases']}</div>
            </div>
        </div>
        """

    html_content += """
    </body>
    </html>
    """

    # Enregistrement physique du document généré
    with open(fichier_out, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n=> 📄 Rapport HTML généré avec succès. Ouvre le fichier : {fichier_out}")
