import pandas as pd
import time
import argparse
from datasets import load_dataset
import google.generativeai as genai

"""
Use Gemini API to rephrase the Klimafakten and Correctiv datasets from 
formal text to tweets. 
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--klimafakten_path", type=str, help="Path for Klimafakten dataset")
    parser.add_argument("--corrective_path", type=str, help="Path for Correctiv dataset")
    parser.add_argument("--google_api_key", type=str, help="Google API to use Gemini")

    args = parser.parse_args()

    klimafakten = pd.read_csv(args.klimafakten_path)

    GOOGLE_API_KEY = args.google_api_key

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')
    
    gemini_rephrasings = []
    
    for idx, row in klimafakten.iterrows():
        print("Rephrasing Klimafakten...")
        claim = row['Fact']
        
        try:
            response = model.generate_content(
               """
               Aufgabe: Gegeben einer Behauptung aus einem Nachrichtenartikel, formuliere diese so um, als wärst du ein Laie, der darüber tweetet.
               
               Einschränkungen:
               1. Beachte stilistische Eigenschaften von Texten aus sozialen Medien, wie beispielsweise die Nutzung von Slang oder informeller Sprache.
               2. Gehe nicht zu weit bei deinen Textgenerierungen. Generiere sie so plausibel wie möglich, sodass man denken könnte, ein Mensch hätte sie geschrieben.
               3. Führe Varianz in die Rhetorik sowie in die syntaktische Struktur deiner Tweets ein. **Nicht jeder Tweet muss eine Frage beinhalten**
               4. **Generiere Tweets in einem neutralen Ton. Verwende keine Ironie oder Satire.**
               5. **Bewahre die wissenschaftliche Behauptung, die in der originalen Behauptung steckt.**
               6. Stelle drei Output-Optionen in einem JSON-Format zur Verfügung, welche eine Liste von Tweets enthält:  {'tweets' : [tweet1, tweet2, tweet3]}.
               7. Bevor du antwortest, formuliere die Prompt um, erweitere die vorliegende Aufgabe und antworte erst danach.
               8. Wenn du aufkommende Fragen hast, generiere sie und beantworte sie, bevor du deinen finalen Output generierst.
               
               Beispieltweets über ein ähnliches Thema:
               1. @KonProg 10% von Schleswig-Holstein sind oder waren Moore. Bewirtschaftet ist die Trocken-Emission bei 75 t pro Jahr und Hektar (100x100m).Größtes Problem wäre der Rückkauf, da sich die meisten in Privatbesitz befinden. (Finanzproblem!)
               2. Wer wissen möchte, wie wir die #Klimakrise ausgelöst und welche politischen und wirtschaftlichen Entscheidungen dazu beigetragen haben, der sollte sich die Doku 'Die Erdzerstörer' ansehen. Vielleicht statt '#Nuhr im Ersten'.
               3. Wenn man sich zu 7. in ein 30 Quadratmeter Passivhaus mit Holz-Solar Heizung quetscht. Keine Verkehrsmittel nutzt, kein Geld für Konsumgüter ausgibt, und sich Bio-Regional-Saisonal-Vegan ernährt, dann ist man im Budget. Ihr wisst was zu tun ist!
               4. @EtzWaerggli @karina_vogel_de Koch in der Geruechtekueche Wer so etwas verbreitet traegt massiv dazu bei, dass wir hier im ganz großen Stil getäuscht, belogen, hinter’s Licht geführt werden. Pfui! Wer Greta finanziert ist gleichgueltig, 97% aller Wissenschaftler teilen ihre Besorgnis.
               5. @BauerWilli_org @NABU_de Lobby d. Agro-Business übt sich in Wagenburg-Mentalität #gruenekreuze: #Insektenschutz, #Dünge-VO, #Klimaschutz – alles wird abgelehnt. Jahrzehntelang wurde verfehlte #Agrarpolitik gegen d. Widerstand d. Gesellschaft forciert u. nun wundert man sich, dass man alleine da steht.
               
               Behauptung: """+ claim + """
               
               Output: """)
            gemini_rephrasings.append(response.text)
        
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            gemini_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")
            
        if (idx+1) % 15 == 0:
            time.sleep(60)
            
    klimafakten['rephrasings'] = gemini_rephrasings
    klimafakten.to_csv('klimafakten_rephrased.csv', index=False)
    print('Finished rephrasing Klimafakten and saved at klimafakten_rephrased.csv')

    correctiv = pd.read_csv(args.corrective_path)

    gemini_rephrasings = []
    for idx, row in correctiv.iterrows():
        claim = row['Behauptung']
        
        try:
            response = model.generate_content(
               """
               Aufgabe: Gegeben einer Behauptung aus einem Nachrichtenartikel, formuliere diese so um, als wärst du ein Laie, der darüber tweetet.
               
               Einschränkungen:
               1. Beachte stilistische Eigenschaften von Texten aus sozialen Medien, wie beispielsweise die Nutzung von Slang oder informeller Sprache.
               2. Gehe nicht zu weit bei deinen Textgenerierungen. Generiere sie so plausibel wie möglich, sodass man denken könnte, ein Mensch hätte sie geschrieben.
               3. Führe Varianz in die Rhetorik sowie in die syntaktische Struktur deiner Tweets ein. **Nicht jeder Tweet muss eine Frage beinhalten**
               4. **Generiere Tweets in einem neutralen Ton. Verwende keine Ironie oder Satire.**
               5. **Bewahre die wissenschaftliche Behauptung, die in der originalen Behauptung steckt.**
               6. Stelle drei Output-Optionen in einem JSON-Format zur Verfügung, welche eine Liste von Tweets enthält:  {'tweets' : [tweet1, tweet2, tweet3]}.
               7. Bevor du antwortest, formuliere die Prompt um, erweitere die vorliegende Aufgabe und antworte erst danach.
               8. Wenn du aufkommende Fragen hast, generiere sie und beantworte sie, bevor du deinen finalen Output generierst.
               
               Beispieltweets über ein ähnliches Thema:
               1. @KonProg 10% von Schleswig-Holstein sind oder waren Moore. Bewirtschaftet ist die Trocken-Emission bei 75 t pro Jahr und Hektar (100x100m).Größtes Problem wäre der Rückkauf, da sich die meisten in Privatbesitz befinden. (Finanzproblem!)
               2. Wer wissen möchte, wie wir die #Klimakrise ausgelöst und welche politischen und wirtschaftlichen Entscheidungen dazu beigetragen haben, der sollte sich die Doku 'Die Erdzerstörer' ansehen. Vielleicht statt '#Nuhr im Ersten'.
               3. Wenn man sich zu 7. in ein 30 Quadratmeter Passivhaus mit Holz-Solar Heizung quetscht. Keine Verkehrsmittel nutzt, kein Geld für Konsumgüter ausgibt, und sich Bio-Regional-Saisonal-Vegan ernährt, dann ist man im Budget. Ihr wisst was zu tun ist!
               4. @EtzWaerggli @karina_vogel_de Koch in der Geruechtekueche Wer so etwas verbreitet traegt massiv dazu bei, dass wir hier im ganz großen Stil getäuscht, belogen, hinter’s Licht geführt werden. Pfui! Wer Greta finanziert ist gleichgueltig, 97% aller Wissenschaftler teilen ihre Besorgnis.
               5. @BauerWilli_org @NABU_de Lobby d. Agro-Business übt sich in Wagenburg-Mentalität #gruenekreuze: #Insektenschutz, #Dünge-VO, #Klimaschutz – alles wird abgelehnt. Jahrzehntelang wurde verfehlte #Agrarpolitik gegen d. Widerstand d. Gesellschaft forciert u. nun wundert man sich, dass man alleine da steht.
               
               Behauptung: """+ claim + """
               
               Output: """)
            gemini_rephrasings.append(response.text)
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            gemini_rephrasings.append(f"Gemini Error: {error_type}, {error_message}")

        if (idx+1) % 15 == 0:
            time.sleep(60)
    
    correctiv['rephrasings'] = gemini_rephrasings
    correctiv.to_csv('corrective_rephrased.csv')
    print('Finished rephrasing Correctiv and saved at corrective_rephrased.csv')


if __name__ == "__main__":
    main()
