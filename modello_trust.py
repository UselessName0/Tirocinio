import numpy as np
import itertools
import json 
import webbrowser # <-- NUOVO IMPORT PER APRIRE IL BROWSER
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ====================================
# FUNZIONI HELPERS E ACQUISIZIONE DATI
# ====================================

stati = ['Basso', 'Medio-basso', 'Neutro', 'Medio-alto', 'Alto']
card = len(stati)

def connetti_google_sheets(url_foglio):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('credenziali.json', scope)
    client = gspread.authorize(creds)
    return client.open_by_url(url_foglio).sheet1

def ottieni_risposte_T0(id_partecipante):
    sheet = connetti_google_sheets("https://docs.google.com/spreadsheets/d/19FiqEfbQGqWZTRgBYmevrhzcSBdt1jh6oquyz8sIMhc/edit?usp=sharing")
    tutti_i_record = sheet.get_all_records()
    
    utente_corretto = None
    for riga in tutti_i_record:
        if str(riga.get("ID Partecipante", "")).strip() == str(id_partecipante).strip():
            utente_corretto = riga
            break
            
    if utente_corretto is None:
        raise ValueError(f"Errore critico: Il partecipante '{id_partecipante}' non è stato trovato nel Google Sheet T0!")
    
    ben_1 = int(utente_corretto["In generale le persone tengono davvero al benessere degli altri"])
    ben_2 = int(utente_corretto["La persona tipica è sinceramente preoccupata per i problemi altrui"])
    ben_3 = int(utente_corretto["La maggior parte delle volte, le persone si preoccupano abbastanza da cercare di essere d'aiuto, piuttosto che pensare solo a sé stesse"])
    media_benevolence = (ben_1 + ben_2 + ben_3) / 3

    int_1 = int(utente_corretto["In generale, la maggior parte delle persone mantiene le proprie promesse"])
    int_2 = int(utente_corretto["Penso che le persone in genere cerchino di sostenere le proprie parole con i fatti"])
    int_3 = int(utente_corretto["La maggior parte delle persone è onesta nei rapporti con gli altri"])
    media_integrity = (int_1 + int_2 + int_3) / 3

    abi_1 = int(utente_corretto["Credo che la maggior parte dei professionisti svolga un ottimo lavoro nel proprio campo"])
    abi_2 = int(utente_corretto["La maggior parte dei professionisti è molto competente nel settore che ha scelto"])
    abi_3 = int(utente_corretto["Una grande magenta di professionisti è competente nella propria area di specializzazione"])
    media_ability = (abi_1 + abi_2 + abi_3) / 3

    prop_1 = int(utente_corretto["Di solito mi fido delle persone finché non mi danno un motivo per non farlo"])
    prop_2 = int(utente_corretto["In genere do alle persone il beneficio del dubbio quando le incontro per la prima volta"])
    prop_3 = int(utente_corretto["Il mio approccio tipico è quello di fidarmi dei nuovi conoscenti finché non dimostrano che non dovrei farlo"])
    media_stance = (prop_1 + prop_2 + prop_3) / 3

    propensity_globale = int(round((media_benevolence + media_integrity + media_ability + media_stance) / 4))
    integrity_iniziale = int(round(media_integrity))

    return {
        'Propensity_to_Trust': propensity_globale,
        'Integrity_Prior': integrity_iniziale, 
        'InitialPropensityRisk': int(utente_corretto["In generale, quanto sei propenso ad accettare un rischio durante l'utilizzo di una nuova tecnologia?"]),
        'Criticality': int(utente_corretto["Quanto consideri critico o pericoloso il compito che il robot sta per svolgere in questo studio?"])
    }

def ottieni_risposte_T3_MID(id_partecipante):
    url_mid = "https://docs.google.com/spreadsheets/d/18qulPTiSaEpmDKAPZFW7hi6s1Ev0kUINDvuKSiHrTfk/edit?usp=sharing"
    sheet = connetti_google_sheets(url_mid)
    tutti_i_record = sheet.get_all_records()
    
    utente = None
    for riga in tutti_i_record:
        if str(riga.get("ID Partecipante", "")).strip() == str(id_partecipante).strip():
            utente = riga
            break
            
    if utente is None:
        raise ValueError(f"Attenzione: Modulo intermedio per '{id_partecipante}' non ancora compilato!")
    
    # ILLEGIBILITA' 
    ill_1 = 6 - int(utente["Il funzionamento generale del robot è un mistero per me"])
    ill_2 = 6 - int(utente["E’ difficile capire il generale funzionamento del robot"])
    ill_3 = 6 - int(utente["E’ difficile avere una chiara visione delle operazioni generali del robot"])
    ill_4 = 6 - int(utente["Sono confuso sul obiettivo generale del robot"])
    ill_5 = 6 - int(utente["Non sono sicuro di cosa faccia il robot"])
    ill_6 = 6 - int(utente["Non capisco quali siano i processi interni del robot"])
    ill_7 = 6 - int(utente["Non so spiegare il comportamento del robot"])
    ill_8 = 6 - int(utente["E’ impossibile sapere cosa il robot faccia"])
    ill_9 = int(utente["Mi è chiaro cosa il robot faccia"])
    ill_10 = int(utente["Ho una chiara comprensione sul come il robot operi in generale"])
    ill_11 = int(utente["Ho l’impressione che le spiegazioni del robot siano utili"])
    media_illegibilita = sum([ill_1, ill_2, ill_3, ill_4, ill_5, ill_6, ill_7, ill_8, ill_9, ill_10, ill_11]) / 11

    # SPIEGABILITA'
    spi_1 = int(utente["Il robot spiega task complessi in un modo semplice da capire"])
    spi_2 = int(utente["Il robot da spiegazioni dettagliate delle sue azioni"])
    spi_3 = int(utente["Il robot da chiare spiegazioni delle sue azioni"])
    spi_4 = int(utente["Le spiegazioni date dal robot sulle sue azioni sono dirette"])
    spi_5 = int(utente["Mi sento informato riguardo le attività del robot"])
    spi_6 = int(utente["Il robot comunica il suo stato generale in maniera effettiva"])
    spi_7 = int(utente["E’ facile per me prevedere le future azioni del robot"])
    media_spiegabilita = sum([spi_1, spi_2, spi_3, spi_4, spi_5, spi_6, spi_7]) / 7

    # PREVEDIBILITA' 
    pre_1 = int(utente["Il comportamento del robot è prevedibile"])
    pre_2 = int(utente["Mi sento sicuro nel predire i movimenti successivi del robot"])
    pre_3 = int(utente["E’ facile anticipare cosa avverrà dal comportamento del robot"])
    pre_4 = 6 - int(utente["E’ difficile per me dire cosa il robot farà successivamente"])
    pre_5 = int(utente["Le prossime azioni del robot sono chiare per me"])
    pre_6 = int(utente["Le azioni del robot sono scontate"])
    pre_7 = int(utente["Il robot da indizi che aiutano a predire le sue successive azioni"])
    pre_8 = 6 - int(utente["Il comportamento del robot non aiuta a predire cosa farà successivamente"])
    media_prevedibilita = sum([pre_1, pre_2, pre_3, pre_4, pre_5, pre_6, pre_7, pre_8]) / 8

    # INTEGRITY
    int_1 = int(utente["In generale, il robot mantiene le proprie promesse"])
    int_2 = int(utente["Penso che il robot in genere cerchi di sostenere le proprie parole con i fatti"])
    int_3 = int(utente["Il robot è stato onesto nei rapporti"])
    media_integrity_mid = (int_1 + int_2 + int_3) / 3

    # BENEVOLENCE (DA AGGIUNGERE)

    # Calcolo Medie Finali Arrotondate
    media_transparency = int(round((media_illegibilita + media_spiegabilita + media_prevedibilita) / 3))
    media_integrity_mid = int(round(media_integrity_mid))

    return {
        'Transparency_Mid': media_transparency,
        'Integrity_Mid': media_integrity_mid
        # Qui in futuro aggiungeremo Benevolence_Mid
    }

def crea_cpt_da_voto(nome_nodo, voto_1_a_5):
    indice_stato = voto_1_a_5 - 1
    probabilita = [0.0] * card
    probabilita[indice_stato] = 1.0
    valori_per_pgmpy = [[p] for p in probabilita]
    return TabularCPD(variable=nome_nodo, variable_card=card, values=valori_per_pgmpy, state_names={nome_nodo: stati})

# ==========================
# CREAZIONE DELLA RETE E CPT
# ==========================

model = DiscreteBayesianNetwork([
    ('TaskSuccess', 'Ability'),
    ('Helpfulness', 'Benevolence'),
    ('Ability', 'Trustworthiness'),
    ('Benevolence', 'Trustworthiness'),
    ('Integrity', 'Trustworthiness'),
    ('Criticality', 'Perceived_Risk'),
    ('Trustworthiness', 'Risk_Taking'),
    ('Perceived_Risk', 'Risk_Taking'),
    ('InitialPropensityRisk', 'Risk_Taking'),
    ('Transparency', 'Trust'), 
    ('Risk_Taking', 'Trust'),
    ('Propensity_to_Trust', 'Trust'),
    ('Trustworthiness', 'Trust')
])

print("="*50)
print("SETUP ESPERIMENTO HRI")
print("="*50)
ID_UTENTE = input("Inserisci l'ID del partecipante per questa sessione (es. P01, P02...): ").strip()

print(f"\n[Attendere...] Connessione a Google Sheets per scaricare i dati T0 di {ID_UTENTE}...")
dati_utente = ottieni_risposte_T0(ID_UTENTE)
print("Dati T0 scaricati con successo! Avvio Rete Bayesiana...")

cpd_task_success = TabularCPD('TaskSuccess', card, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={'TaskSuccess': stati})
cpd_helpfulness = TabularCPD('Helpfulness', card, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={'Helpfulness': stati})
cpd_transparency = TabularCPD('Transparency', card, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={'Transparency': stati})

cpd_integrity = crea_cpt_da_voto('Integrity', dati_utente['Integrity_Prior'])
cpd_prop_trust = crea_cpt_da_voto('Propensity_to_Trust', dati_utente['Propensity_to_Trust'])
cpd_initial_risk = crea_cpt_da_voto('InitialPropensityRisk', dati_utente['InitialPropensityRisk'])
cpd_criticality = crea_cpt_da_voto('Criticality', dati_utente['Criticality'])

def genera_cpt_un_genitore(nome_nodo, nome_genitore):
    matrice_logica = [
        [0.70, 0.20, 0.05, 0.00, 0.00], [0.20, 0.60, 0.15, 0.05, 0.00], [0.10, 0.15, 0.60, 0.15, 0.10], 
        [0.00, 0.05, 0.15, 0.60, 0.20], [0.00, 0.00, 0.05, 0.20, 0.70]  
    ]
    return TabularCPD(variable=nome_nodo, variable_card=card, values=matrice_logica, evidence=[nome_genitore], evidence_card=[card], state_names={nome_nodo: stati, nome_genitore: stati})

cpd_ability = genera_cpt_un_genitore('Ability', 'TaskSuccess')
cpd_benevolence = genera_cpt_un_genitore('Benevolence', 'Helpfulness')
cpd_perceived_risk = genera_cpt_un_genitore('Perceived_Risk', 'Criticality')

def genera_cpt_multi(nome_nodo, nomi_genitori, pesi=None, direzioni=None):
    num_genitori = len(nomi_genitori)
    if pesi is None: pesi = [1.0] * num_genitori
    if direzioni is None: direzioni = [1] * num_genitori 
    pesi_norm = [p / sum(pesi) for p in pesi]
    matrice = [[] for _ in range(card)]
    combinazioni = list(itertools.product(range(card), repeat=num_genitori))
    
    for combo in combinazioni:
        punteggio_totale = sum((((card - 1) - combo[i]) if direzioni[i] == -1 else combo[i]) * pesi_norm[i] for i in range(num_genitori))
        prob_colonna = [np.exp(-(abs(stato_figlio - punteggio_totale)**2) / 0.15) for stato_figlio in range(card)]
        somma_prob = sum(prob_colonna)
        for riga in range(card): matrice[riga].append(prob_colonna[riga] / somma_prob)
            
    return TabularCPD(variable=nome_nodo, variable_card=card, values=matrice, evidence=nomi_genitori, evidence_card=[card]*num_genitori, state_names={nome_nodo: stati, **{g: stati for g in nomi_genitori}})

cpd_trustworthiness = genera_cpt_multi('Trustworthiness', ['Ability', 'Benevolence', 'Integrity'], direzioni=[1, 1, 1], pesi=[3.0, 3.0, 1.0])
cpd_risk_taking = genera_cpt_multi('Risk_Taking', ['Trustworthiness', 'Perceived_Risk', 'InitialPropensityRisk'], direzioni=[1, -1, 1], pesi=[4.0, 1.5, 1.0])
cpd_trust = genera_cpt_multi('Trust', ['Risk_Taking', 'Propensity_to_Trust', 'Trustworthiness', 'Transparency'], direzioni=[1, 1, 1, 1], pesi=[1.0, 0.5, 4.0, 1.5])

model.add_cpds(cpd_task_success, cpd_helpfulness, cpd_transparency, cpd_criticality, cpd_initial_risk, cpd_prop_trust, cpd_integrity, 
               cpd_ability, cpd_benevolence, cpd_perceived_risk, cpd_trustworthiness, cpd_risk_taking, cpd_trust)

# ================================
# CARICAMENTO SCENARIO JSON E LOOP
# ================================
with open('scenario.json', 'r', encoding='utf-8') as f:
    interazioni_robot = json.load(f)

print("\n" + "="*50)
print("INIZIO ESPERIMENTO")
print("="*50)

infer = VariableElimination(model)

print("\n--- ROUND 0: Baseline Utente (Pre-interazione) ---")
risultato_t0 = infer.query(variables=['Trust'])
print(f"-> Livello di TRUST iniziale: {stati[np.argmax(risultato_t0.values)]} ({(np.max(risultato_t0.values)*100):.1f}%)")

nodi_da_monitorare = ['Ability', 'Trustworthiness', 'Risk_Taking']

for step in interazioni_robot:
    t = step['Time']
    print(f"\n--- ROUND {t}: {step['Scenario']} ---")
    print(f"Evidenze Robot: {step['Evidenze']}")
    
    print("  [Dinamiche Interne]:")
    for nodo in nodi_da_monitorare:
        res_nodo = infer.query(variables=[nodo], evidence=step['Evidenze'])
        print(f"    - {nodo}: {stati[np.argmax(res_nodo.values)]} ({(np.max(res_nodo.values)*100):.1f}%)")
    
    risultato = infer.query(variables=['Trust'], evidence=step['Evidenze'])
    probabilita_attuali = risultato.values
    print(f"-> Livello di TRUST calcolato dalla Rete: {stati[np.argmax(probabilita_attuali)]} ({(np.max(probabilita_attuali)*100):.1f}%)")
    
    # Aggiornamento Rolling Prior Propensity
    valori_nuovi = [[p] for p in probabilita_attuali]
    nuova_cpt_propensity = TabularCPD('Propensity_to_Trust', card, values=valori_nuovi, state_names={'Propensity_to_Trust': stati})
    model.remove_cpds('Propensity_to_Trust')
    model.add_cpds(nuova_cpt_propensity)
    infer = VariableElimination(model)

    # ==================================
    # PAUSA PER IL QUESTIONARIO DI MEZZO
    # ==================================
    if t == 3:
        print("\n" + "!"*50)
        print("MOMENTO CRITICO: Apertura automatica del questionario di mezzo.")
        print("!"*50)
        url_link_form_mid = "https://forms.gle/3b2NV8pDQDjJdhFj8"
        
        # Questa riga apre automaticamente il browser
        webbrowser.open(url_link_form_mid)
        
        input("Premi [INVIO] non appena l'utente ha cliccato 'Invia' sul Google Form MID...")
        
        print("\nLettura dei dati MID da Google Sheets in corso...")
        try:
            dati_mid = ottieni_risposte_T3_MID(ID_UTENTE)
            print(f"  * Trasparenza Calcolata dal MID: {stati[dati_mid['Transparency_Mid']-1]}")
            print(f"  * Integrity Calcolata dal MID: {stati[dati_mid['Integrity_Mid']-1]}")
            
            # Aggiornamento nodi con i dati reali del questionario di mezzo
            nuova_cpt_transparency = crea_cpt_da_voto('Transparency', dati_mid['Transparency_Mid'])
            nuova_cpt_integrity = crea_cpt_da_voto('Integrity', dati_mid['Integrity_Mid'])
            # nuova_cpt_benevolence = crea_cpt_da_voto('Benevolence', dati_mid['Benevolence_Mid'])  # Da implementare quando ci saranno i dati
            
            model.remove_cpds('Transparency', 'Integrity')
            model.add_cpds(nuova_cpt_transparency, nuova_cpt_integrity)
            infer = VariableElimination(model)
            
            print("  [INFO] I Nodi Radice 'Transparency' e 'Integrity' sono stati aggiornati per i Round 4-6.")
            
        except Exception as e:
            print(f"Impossibile leggere i dati medi o aggiornare la rete: {e}")
            
        print("\nRipresa dell'esperimento...")
        print("="*50)