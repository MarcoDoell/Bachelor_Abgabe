import pandas as pd 

df = pd.read_csv(file_path ,sep=';',header=None, encoding='ISO-8859-1') 
do_preprocessing(df)


def do_preprocessing(df)

    # Filtern der Spalten (Schritt 1)
    df= df[['KtoNr','Umsatz_S', 'Umsatz_H', 'Anwender']]

    # Handle missing values (Schritt 2)
    df['KtoNr'].dropna(inplace=True)
    df['Umsatz_S'].fillna(0, inplace=True)
    df['Umsatz_H'].fillna(0, inplace=True)
    df['Anwender'].fillna('Unbekannt', inplace=True)

    # Filtern der Kontonummern für den Kontenbereich Umsatzerlöse (Schritt 3)
    df = df[((df['KtoNr'] >= 8000) & (df['KtoNr'] <= 8589)) | ((df['KtoNr'] >= 8900) 
         & (df['KtoNr'] <= 8919)) | ((df['KtoNr'] >= 8940) & (df['KtoNr'] <= 8959)) 
         | ((df['KtoNr'] >= 8700) & (df['KtoNr'] <= 8799))].copy()

    # Datentyptransformationen (Schritt 4)
    df['Umsatz_H'] = df['Umsatz_H'].str.replace(',','.')
    df['Umsatz_H'] = pd.to_numeric(df['Umsatz_H'])

    df['Umsatz_S'] = df['Umsatz_S'].str.replace(',','.')
    df['Umsatz_S'] = pd.to_numeric(df['Umsatz_S'])

    df['Anwender'] = df.Anwender.astype('category')
    df['Anwender'] = df['Anwender'].cat.codes

    # Berechnung des Saldos (Schritt 5)
    df['Saldo'] = df.apply(lambda row: row.Umsatz_H - row.Umsatz_S, axis = 1) 

    # Entfernen nicht mehr benötigter Spalten (Schritt 6)
    df = df.drop(columns=['Umsatz_S','Umsatz_H', 'KtoNr'])


    return df