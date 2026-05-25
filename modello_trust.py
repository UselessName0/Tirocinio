import numpy as np
import itertools
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

def ottieni_risposte_da_google_sheets(id_partecipante):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('credenziali.json', scope)
    client = gspread.authorize(creds)
    
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/19FiqEfbQGqWZTRgBYmevrhzcSBdt1jh6oquyz8sIMhc/edit?usp=sharing").sheet1
    
    tutti_i_record = sheet.get_all_records()
    
    utente_corretto = None
    for riga in tutti_i_record:
        if str(riga.get("ID Partecipante", "")).strip() == str(id_partecipante).strip():
            utente_corretto = riga
            break
            
    if utente_corretto is None:
        raise ValueError(f"Errore critico: Il partecipante '{id_partecipante}' non è stato trovato nel Google Sheet!")
    
    # --- BENEVOLENCE ---
    ben_1 = int(utente_corretto["In generale le persone tengono davvero al benessere degli altri"])
    ben_2 = int(utente_corretto["La persona tipica è sinceramente preoccupata per i problemi altrui"])
    ben_3 = int(utente_corretto["La maggior parte delle volte, le persone si preoccupano abbastanza da cercare di essere d'aiuto, piuttosto che pensare solo a sé stesse"])
    media_benevolence = (ben_1 + ben_2 + ben_3) / 3

    # --- INTEGRITY ---
    int_1 = int(utente_corretto["In generale, la maggior parte delle persone mantiene le proprie promesse"])
    int_2 = int(utente_corretto["Penso che le persone in genere cerchino di sostenere le proprie parole con i fatti"])
    int_3 = int(utente_corretto["La maggior parte delle persone è onesta nei rapporti con gli altri"])
    media_integrity = (int_1 + int_2 + int_3) / 3

    # --- ABILITY ---
    abi_1 = int(utente_corretto["Credo che la maggior parte dei professionisti svolta un ottimo lavoro nel proprio campo"])
    abi_2 = int(utente_corretto["La maggior parte dei professionisti è molto competente nel settore che ha scelto"])
    abi_3 = int(utente_corretto["Una grande maggioranza di professionisti è competente nella propria area di specializzazione"])
    media_ability = (abi_1 + abi_2 + abi_3) / 3

    # --- PROPENSITY TO TRUST  ---
    prop_1 = int(utente_corretto["Di solito mi fido delle persone finché non mi danno un motivo per non farlo"])
    prop_2 = int(utente_corretto["In genere do alle persone il beneficio del dubbio quando le incontro per la prima volta"])
    prop_3 = int(utente_corretto["Il mio approccio tipico è quello di fidarmi dei nuovi conoscenti finché non dimostrano che non dovrei farlo"])
    media_stance = (prop_1 + prop_2 + prop_3) / 3

    # La propensione generale è la media delle 4 dimensioni (arrotondata all'intero più vicino per la CPT 1-5)
    propensity_globale = int(round((media_benevolence + media_integrity + media_ability + media_stance) / 4))
    
    # L'integrity iniziale (per il nodo radice) è l'arrotondamento della sua specifica dimensione
    integrity_iniziale = int(round(media_integrity))

    risposte = {
        'Propensity_to_Trust': propensity_globale,
        'Integrity_Prior': integrity_iniziale, 
        'InitialPropensityRisk': int(utente_corretto["In generale, quanto sei propenso ad accettare un rischio durante l'utilizzo di una nuova tecnologia?"]),
        'Criticality': int(utente_corretto["Quanto consideri critico o pericoloso il compito che il robot sta per svolgere in questo esperimento?"])
    }
    
    return risposte

def crea_cpt_da_voto(nome_nodo, voto_1_a_5):
    indice_stato = voto_1_a_5 - 1
    probabilita = [0.0] * card
    probabilita[indice_stato] = 1.0
    valori_per_pgmpy = [[p] for p in probabilita]
    return TabularCPD(variable=nome_nodo, variable_card=card, values=valori_per_pgmpy, state_names={nome_nodo: stati})

# ====================
# CREAZIONE DELLA RETE
# ====================

model = DiscreteBayesianNetwork([
    ('TaskSuccess', 'Ability'),
    ('Helpfulness', 'Benevolence'),
    ('leggibility', 'Transparency'),
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

dati_utente = ottieni_risposte_da_google_sheets("P00") 

# ==============================
# CREAZIONE TABELLE CPT (RADICE)
# ==============================

cpd_task_success = TabularCPD('TaskSuccess', card, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={'TaskSuccess': stati})
cpd_helpfulness = TabularCPD('Helpfulness', card, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={'Helpfulness': stati})
cpd_leggibility = TabularCPD('leggibility', card, values=[[0.2], [0.2], [0.2], [0.2], [0.2]], state_names={'leggibility': stati})

cpd_integrity = crea_cpt_da_voto('Integrity', dati_utente['Integrity_Prior'])
cpd_prop_trust = crea_cpt_da_voto('Propensity_to_Trust', dati_utente['Propensity_to_Trust'])
cpd_initial_risk = crea_cpt_da_voto('InitialPropensityRisk', dati_utente['InitialPropensityRisk'])
cpd_criticality = crea_cpt_da_voto('Criticality', dati_utente['Criticality'])

# ==================================
# CREAZIONE TABELLE CPT (NODI FIGLI)
# ==================================

def genera_cpt_un_genitore(nome_nodo, nome_genitore):
    matrice_logica = [
        [0.70, 0.20, 0.05, 0.00, 0.00], 
        [0.20, 0.60, 0.15, 0.05, 0.00], 
        [0.10, 0.15, 0.60, 0.15, 0.10], 
        [0.00, 0.05, 0.15, 0.60, 0.20], 
        [0.00, 0.00, 0.05, 0.20, 0.70]  
    ]
    return TabularCPD(variable=nome_nodo, variable_card=card, values=matrice_logica, 
                      evidence=[nome_genitore], evidence_card=[card], state_names={nome_nodo: stati, nome_genitore: stati})

cpd_ability = genera_cpt_un_genitore('Ability', 'TaskSuccess')
cpd_benevolence = genera_cpt_un_genitore('Benevolence', 'Helpfulness')
cpd_transparency = genera_cpt_un_genitore('Transparency', 'leggibility')
cpd_perceived_risk = genera_cpt_un_genitore('Perceived_Risk', 'Criticality')

def genera_cpt_multi(nome_nodo, nomi_genitori, pesi=None, direzioni=None):
    num_genitori = len(nomi_genitori)
    if pesi is None: pesi = [1.0] * num_genitori
    if direzioni is None: direzioni = [1] * num_genitori 
    pesi_norm = [p / sum(pesi) for p in pesi]
    matrice = [[] for _ in range(card)]
    combinazioni = list(itertools.product(range(card), repeat=num_genitori))
    
    for combo in combinazioni:
        punteggio_totale = 0.0
        for i in range(num_genitori):
            stato_gen = combo[i]
            if direzioni[i] == -1: stato_gen = (card - 1) - stato_gen
            punteggio_totale += stato_gen * pesi_norm[i]
            
        prob_colonna = []
        for stato_figlio in range(card):
            distanza = abs(stato_figlio - punteggio_totale)
            prob = np.exp(-(distanza**2) / 0.15) 
            prob_colonna.append(prob)
            
        somma_prob = sum(prob_colonna)
        for riga in range(card): matrice[riga].append(prob_colonna[riga] / somma_prob)
            
    return TabularCPD(variable=nome_nodo, variable_card=card, values=matrice,
                      evidence=nomi_genitori, evidence_card=[card]*num_genitori, state_names={nome_nodo: stati, **{g: stati for g in nomi_genitori}})

cpd_trustworthiness = genera_cpt_multi('Trustworthiness', ['Ability', 'Benevolence', 'Integrity'], direzioni=[1, 1, 1], pesi=[3.0, 3.0, 1.0])
cpd_risk_taking = genera_cpt_multi('Risk_Taking', ['Trustworthiness', 'Perceived_Risk', 'InitialPropensityRisk'], direzioni=[1, -1, 1], pesi=[4.0, 1.5, 1.0])
cpd_trust = genera_cpt_multi('Trust', ['Risk_Taking', 'Propensity_to_Trust', 'Trustworthiness', 'Transparency'], direzioni=[1, 1, 1, 1], pesi=[1.0, 0.5, 4.0, 1.5])

model.add_cpds(cpd_task_success, cpd_helpfulness, cpd_leggibility, cpd_criticality, cpd_initial_risk, cpd_prop_trust, cpd_integrity, 
               cpd_ability, cpd_benevolence, cpd_transparency, cpd_perceived_risk, cpd_trustworthiness, cpd_risk_taking, cpd_trust)

# =======================
# SIMULAZIONE ESPERIMENTO 
# =======================
print("\n" + "="*25)
print("INIZIO ESPERIMENTO")
print("="*25)

infer = VariableElimination(model)

storico_probabilita = {stato: [] for stato in stati} 
tempi = []

print("\n--- ROUND 0: Baseline Utente (Pre-interazione) ---")
risultato_t0 = infer.query(variables=['Trust'])
prob_t0 = risultato_t0.values
stato_vinc_t0 = stati[np.argmax(prob_t0)]
print(f"-> Livello di TRUST iniziale (solo questionario): {stato_vinc_t0} ({(np.max(prob_t0)*100):.1f}%)")

tempi.append("T0")
for i, stato in enumerate(stati):
    storico_probabilita[stato].append(prob_t0[i] * 100)

interazioni_robot = [
    {"Time": 1, "Scenario": "Il robot fa un lavoro perfetto e chiaro",
     "Evidenze": {'TaskSuccess': 'Alto', 'Helpfulness': 'Alto', 'leggibility': 'Medio-alto'}},
    {"Time": 2, "Scenario": "Il robot commette un errore grave",
     "Evidenze": {'TaskSuccess': 'Basso', 'Helpfulness': 'Basso', 'leggibility': 'Basso'}},
    {"Time": 3, "Scenario": "Il robot riprova, successo medio",
     "Evidenze": {'TaskSuccess': 'Medio-alto', 'Helpfulness': 'Neutro', 'leggibility': 'Neutro'}}
]

nodi_da_monitorare = ['Ability', 'Trustworthiness', 'Risk_Taking']

for step in interazioni_robot:
    t = step['Time']
    print(f"\n--- ROUND {t}: {step['Scenario']} ---")
    print(f"Dati Sensori (Evidenze): {step['Evidenze']}")
    
    print("  [Dinamiche Interne della Rete]:")
    for nodo in nodi_da_monitorare:
        res_nodo = infer.query(variables=[nodo], evidence=step['Evidenze'])
        stato_max = stati[np.argmax(res_nodo.values)]
        print(f"    - {nodo}: {stato_max} ({(np.max(res_nodo.values)*100):.1f}%)")
    
    risultato = infer.query(variables=['Trust'], evidence=step['Evidenze'])
    probabilita_attuali = risultato.values
    
    stato_vincente = stati[np.argmax(probabilita_attuali)]
    percentuale_vincente = np.max(probabilita_attuali) * 100
    print(f"-> Livello di TRUST Finale calcolato: {stato_vincente} ({percentuale_vincente:.1f}%)")
    
    tempi.append(f"T{t}")
    for i, stato in enumerate(stati):
        storico_probabilita[stato].append(probabilita_attuali[i] * 100) 
    
    valori_nuovi = [[p] for p in probabilita_attuali]
    nuova_cpt_propensity = TabularCPD('Propensity_to_Trust', card, values=valori_nuovi, state_names={'Propensity_to_Trust': stati})
    
    model.remove_cpds('Propensity_to_Trust')
    model.add_cpds(nuova_cpt_propensity)
    infer = VariableElimination(model)

print("\n" + "="*25)