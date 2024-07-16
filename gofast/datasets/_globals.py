# -*- coding: utf-8 -*-
"""
`_globals.py` module designed for internal use within :code:`gofast` package, 
containing global variables, constants, and shared configurations. This module
typically includes definitions that are used across multiple `datasets` module 
ensuring consistency and ease of maintenance. The leading underscore in its 
name implies that it's not intended for external use but rather serves as a 
supportive component within the package.

"""
DYSPNEA_DICT = {
    'start_date': 'The date when the observation or data collection period began.',
    'starttime': 'The time when the observation or data collection period started.',
    'endtime': 'The time when the observation or data collection period ended.',
    'date_entered': 'The date when the data was entered into the dataset.',
    'submission_time': 'The time when the data was submitted for inclusion in the dataset.',
    'version': 'The version of the data collection form or dataset structure.',
    'submitted_by': 'The individual or entity that submitted the data.',
    'meta_instanceid': 'A unique identifier for the instance of data collection.',
    'uuid': 'A universally unique identifier for the record.',
    'id': 'A unique identifier for the patient or observation.',
    'gender': 'The gender of the patient (e.g., male, female), as self-identified by the patient.',
    'index': 'A sequential number or index assigned to the observation.',
    'file_number': 'A unique file number assigned to the patient’s record.',
    'age': 'The age of the patient at the time of observation.',
    'not': 'Possibly a field for notes or remarks (requires clarification).',
    'pad': 'Peripheral artery disease presence or assessment.',
    'fc': 'Functional class; a measure of the severity of symptoms.',
    'fr': 'Respiratory rate; the number of breaths per minute.',
    'spo': 'Oxygen saturation; a measure of the amount of oxygen carried in the blood.',
    'temperature': 'The patient’s body temperature.',
    'glasgow_score': 'The Glasgow Coma Scale score, assessing the consciousness level.',
    'diagnosis_pneumonitis': 'Indicator of whether pneumonitis was diagnosed.',
    'diagnosis_asthma_attack': 'Indicator of whether an asthma attack was diagnosed.',
    'diagnosis_pulmonary_tuberculosis': 'Indicator of whether pulmonary tuberculosis was diagnosed.',
    'diagnosis_covid_19': 'Indicator of whether COVID-19 was diagnosed.',
    'diagnosis_heart_failure': 'Indicator of whether heart failure was diagnosed.',
    'diagnosis_copd': 'Indicator of whether chronic obstructive pulmonary disease (COPD) was diagnosed.',
    'diagnosis_bronchial_cancer': 'Indicator of whether bronchial cancer was diagnosed.',
    'diagnosis_pulmonary_fibrosis': 'Indicator of whether pulmonary fibrosis was diagnosed.',
    'diagnostic_other': 'Field for other diagnoses not specifically listed.',
    'parent_index': 'Link or reference to a parent record or observation, if applicable.',
    'duration': 'The duration of the dyspnea episode or symptoms.',
    'xform_id': 'A form or transformation identifier related to data processing.',
    'dyspnea': 'The presence and severity of dyspnea or difficulty breathing.',
    'nyha_intensity': 'The New York Heart Association classification for the intensity of heart failure symptoms.',
    'frequency': 'The frequency of symptoms or episodes.',
    'cough': 'The presence and characteristics of cough.',
    'fever': 'The presence of fever.',
    'asthenia': 'The presence of asthenia or abnormal physical weakness or lack of energy.',
    'admission_method': 'The method or reason for admission to the healthcare facility.',
    'establishment_of_origin': 'The originating establishment or location of the patient before admission.',
    'toxic_tobacco': 'Tobacco use or exposure assessment.',
    'toxic_alcohol': 'Alcohol use or exposure assessment.',
    'condition': 'The general condition or status of the patient.',
    'state_of_the_pupils': 'Assessment of the pupils’ condition or reactivity.',
    'conjunctivas': 'Assessment of the conjunctivas, indicating potential anemia or jaundice.',
    'imo': 'Possibly a specific medical observation or indicator (requires clarification).',
    'condition_of_the_mucous_membranes': 'The condition or appearance of mucous membranes.',
    'dehydration_skin_fold': 'Assessment of dehydration through skin fold test.',
    'respiratory_distress': 'The presence and severity of respiratory distress.',
    'heart_sound': 'Assessment of heart sounds, indicating potential cardiac issues.',
    'breath': 'Characteristics of breathing or breath sounds.',
    'heart_failure': 'Indicator or assessment of heart failure.',
    'lymphadenopathy': 'The presence of enlarged lymph nodes.',
    'diagnosis_retained': 'Final diagnosis or retained diagnosis after assessment.',
    'outcome_of_hospitalization': 'The outcome following hospitalization (e.g., discharged, transferred, deceased).',
}
DYSPNEA_LABELS_DESCR={
    'gender': {
        0: 'Female',
        1: 'Male'},
     'dyspnea': {
         0: 'Acute',
         1: 'Chronic'},
     'nyha_intensity': {
         2: 'IV', 
         1: 'III', 
         0: 'II'},
     'frequency': {
         1: 'Polypnea',
         0: 'Bradypnea'},
     'cough': {
         1: 'Yes', 
         0: 'No'},
     'fever': {
         0: 'No',
         1: 'Yes'},
     'asthenia': {
         0: 'No',
         1: 'Yes'},
     'admission_method': {
         4: 'Taxi',
         0: 'Ambulance',
         2: 'Personal vehicle',
         1: 'Firefighter',
         3: 'SAMU'},
     'establishment_of_origin': {
         4: 'Clinical',
        13: 'Residence',
        11: 'ICA',
        10: 'HMA',
        7: 'HG',
        1: 'CAT',
        12: 'PMI',
        9: 'HG Sikensi',
        8: 'HG Adjamé',
        3: 'Clinic and',
        6: 'General Hospital',
        0: 'Ambulance',
        2: 'CHR',
        14: 'University Hospital',
        5: 'FSU',
        15: 'Yopougon'},
     'toxic_tobacco': {
         0: 'No', 
         1: 'Yes'},
     'toxic_alcohol': {
         0: 'No',
         1: 'Yes'},
     'condition': {
         1: 'Bad', 
         0: 'Average', 
         2: 'Good'},
     'state_of_the_pupils': {
         0: 'Normal', 
         1: 'Unnatural'},
     'conjunctivas': {
         1: 'Colorful', 
         0: 'Blades',
         2: 'Not very colorful'},
     'imo': {
         1: 'Yes',
         0: 'No'},
     'condition_of_the_mucous_membranes': {
         0: 'Dry',
         1: 'Wet'},
     'dehydration_skin_fold': {
         1: 'Yes', 
         0: 'No'},
     'respiratory_distress': {
         1: 'Yes', 
         0: 'No'},
     'heart_sound': {
         0: 'Normal', 
         1: 'Unnatural'},
     'breath': {
         0: 'No',
         1: 'Yes'},
     'heart_failure': {
         0: 'No', 
         1: 'Yes'},
     'lymphadenopathy': {
         0: 'No', 
         1: 'Yes'},
     'diagnosis_retained': {
          17: 'Pneumonitis',
          13: 'Other',
          16: 'Pneumonia Other',
          3: 'Covid 19',
          11: 'Heart failure',
          2: 'COPD',
          9: 'Covid 19 pneumonia',
          23: 'Pulmonary tuberculosis Other',
          19: 'Pulmonary tuberculosis',
          0: 'Asthma attack',
          18: 'Pneumonitis Pulmonary tuberculosis',
          14: 'Pneumonia Asthma attack',
          26: 'pulmonary fibrosis Other',
          10: 'Covid 19 pneumonia Other',
          7: 'Covid 19 Other',
          12: 'Heart failure Other',
          8: 'Covid 19 asthma attack',
          1: 'Bronchial cancer',
          6: 'Covid 19 Heart failure',
          22: 'Pulmonary tuberculosis Covid 19 Other',
          15: 'Pneumonia Heart failure',
          24: 'Pulmonary tuberculosis pulmonary fibrosis Other',
          25: 'pulmonary fibrosis',
          21: 'Pulmonary tuberculosis Covid 19',
          20: 'Pulmonary tuberculosis COPD',
          5: 'Covid 19 COPD',
          4: 'Covid 19 Bronchial cancer'},
     'outcome_of_hospitalization': {
          2: 'Return home',
          0: 'Deceased',
          3: 'Transfer to another care unit',
          1: 'Discharge against medical advice'}
     }

FORENSIC_BF_DICT={
    'timestamp': 'Timestamp',
    'gender': 'sex',
    'age': 'Age',
    'education_level': 'Level of study',
    'occupation': 'Occupation',
    'dna_knowledge': 'Do you think you know enough about using DNA to solve crimes?',
    'dna_info_source': 'If YES, where did you get this information about using DNA to solve crimes?',
    'support_national_dna_db_bf': 'As part of criminal investigations: Do you think that the creation of a national DNA database in Burkina Faso is:',
    'dna_db_custodian_bf': 'Who should be responsible for the custody and management of a national DNA database in Burkina Faso?',
    'dna_db_inclusion_criteria': 'Criteria for inclusion of a genetic profile in a DNA database. To be reserved for:',
    'include_crime_scene_profiles': 'Should profiles from crime scenes be included directly in the national DNA database?',
    'offense_type_dna_recording': 'What type of offense would merit the DNA profile of a convicted person being recorded in the database?',
    'dna_storage_duration': 'For how long do you consider it necessary or normal for a DNA profile to be stored in a national database?',
    'dna_use_family_research': 'As part of family research',
    'dna_use_disaster_research': 'As part of research in the event of natural disasters and attacks',
    'dna_use_interpol_cooperation': 'As part of Cooperation with INTERPOL',
    'dna_use_terrorism_fight': 'As part of the fight against terrorism and organized crime',
    'privacy_invasion_opinion': 'Do you think this is an invasion of privacy?',
    'voluntary_dna_donation': 'Would you agree to voluntarily donate your own DNA to enrich a possible genetic database? (NB: This could help, for example, to find your loved ones or to identify you in the event of your disappearance...)',
    'privacy_risk_concern': 'What is your level of concern about the risk of invasion of privacy?',
    'database_misuse_concern': 'Are you concerned about misuse of this database?',
    'dna_use_in_investigations': 'Do you think they use DNA profiles in criminal investigations?',
    'police_lab_support_need': 'Do you think that the police and national gendarmerie services must be equipped with scientific and especially genetic laboratories to support criminal investigations?',
    'forensic_dna_private_sector': 'Do you instead think that forensic DNA testing should be carried out by the private sector?',
    'forensic_dna_autonomous_institution': 'Or do you rather think that forensic DNA testing should be carried out by an autonomous state institution other than the Police and Gendarmerie?',
    'message_to_investigators': 'What would you like to say to the initiators of this investigation?'
}

FORENSIC_LABELS_DESCR={
 'gender': {
                  1: 'Male', 
                  0: 'Female'},
 'age': {
                  1: '35-60', 
                  0: '16-35', 
                  2: '>60'},
 'education_level': {
                  0: 'PhD',
                  2: 'University',
                  1: 'Secondary'},
 'occupation': {
                  5: 'Ministry responsible for security',
                  7: 'Students',
                  6: 'Others',
                  1: 'Ministry of Higher Education and Scientific Research, Others',
                  0: 'Ministry of Higher Education and Scientific Research',
                  3: 'Ministry of Justice',
                  2: 'Ministry of Higher Education and Scientific Research, Students',
                  8: 'Students, Others',
                  4: 'Ministry of Justice, Others'},
 'dna_knowledge': {
                  3: 'Yes', 
                  0: 'I am not sure', 
                  1: "I don't know", 
                  2: 'No'},
 'dna_info_source': {0: 'During my studies (secondary and/or higher)',
                  4: 'Reading (newspapers and scientific documents)',
                  5: 'The media (TV news, radio, documentaries)',
                  3: 'Others',
                  1: 'Fiction films',
                  2: 'NaN'},
 'support_national_dna_db_bf': {
                  0: 'Important',
                  3: 'Unitile',
                  2: 'Not very important',
                  1: 'Ineffective'},
 'dna_db_custodian_bf': {
                  1: 'Ministry of Justice',
                  4: 'The national police and gendarmerie services',
                  3: 'Others',
                  2: 'Ministry responsible for security',
                  0: 'An autonomous institution'},
 'dna_db_inclusion_criteria': {
                  0: 'Condemned',
                  3: 'Entire population of Burkina Faso',
                  2: 'Convicts, suspects and volunteers',
                  1: 'Convicts and suspects',
                  4: 'Nobody'},
 'include_crime_scene_profiles': {
                  1: 'No',
                  2: 'Yes',
                  0: "I don't know",
                  3: 'am indifferent'},
 'offense_type_dna_recording': {
                  0: 'All crimes',
                  1: 'All crimes and offenses',
                  2: 'Gender-based violence',
                  4: 'Serious crimes only',
                  3: 'No answer'},
 'dna_storage_duration': { 
                  0: 'Indefinitely',
                  2: 'Until acquittal',
                  3: 'Until the death of the condemned',
                  1: 'No answer'},
 'dna_use_family_research': {
                  1: 'No I do not agree',
                  3: 'Yes, I agree but in certain circumstances',
                  2: 'Yes I agree',
                  0: "I don't know"},
 'dna_use_disaster_research': { 
                  3: 'Yes, I agree but in certain circumstances',
                  2: 'Yes I agree',
                  0: "I don't know",
                  1: 'No I do not agree'},
 'dna_use_interpol_cooperation': {
                  2: 'Yes I agree',
                  3: 'yes I agree but in certain circumstances',
                  0: "I don't know",
                  1: 'No I do not agree'},
 'dna_use_terrorism_fight': {
                  2: 'Yes I agree',
                  3: 'yes I agree but in certain circumstances',
                  0: "I don't know",
                  1: 'No I do not agree'},
 'privacy_invasion_opinion': {
                  1: 'No',
                  0: 'Maybe', 
                  2: 'Yes'},
 'voluntary_dna_donation': {
                  3: 'Yes, but under certain conditions',
                  2: 'Yes',
                  1: 'No',
                  0: 'Maybe'},
 'privacy_risk_concern': {
                  4: 'No problem',
                  2: 'Important concerns',
                  3: 'Minor concerns',
                  0: "I don't know",
                  5: 'am indifferent',
                  1: 'I have no response'},
 'database_misuse_concern': {
                  3: 'Minor concerns',
                  1: 'Important concerns',
                  5: 'No problem',
                  2: 'Indifferent',
                  4: 'No answer',
                  0: "I don't know"},
 'dna_use_in_investigations': {
                  2: 'No',
                  1: 'Maybe',
                  0: "I don't know",
                  3: 'Yes'},
 'police_lab_support_need': {
                  2: 'Yes', 
                  1: 'No', 
                  0: 'Maybe'},
 'forensic_dna_private_sector': {
                  1: 'Maybe',
                  3: 'Yes',
                  2: 'No',
                  0: "I don't know"},
 'forensic_dna_autonomous_institution': {
                  3: 'Yes',
                  2: 'No',
                  1: 'Maybe',
                  0: "I don't know"}
 }  
    

WATER_QUAN_NEEDS= {
    "Agri Demand": "Agricultural Water Demand",
    "Indus Demand": "Industrial Water Demand",
    "Domestic Demand": "Domestic Water Demand",
    "Municipal Demand": "Municipal Water Demand",
    "Livestock Needs": "Livestock Water Needs",
    "Irrigation Req": "Irrigation Water Requirements",
    "Hydropower Gen": "Hydropower Generation",
    "Aquaculture Usage": "Aquaculture Water Usage",
    "Mining Consumption": "Mining Water Consumption",
    "Thermal Plant Consumption": "Thermal Power Plant Water Consumption",
    "Ecosystems": "Water for Ecosystems",
    "Forestry": "Water for Forestry",
    "Recreation": "Water for Recreation",
    "Urban Dev": "Water for Urban Development",
    "Drinking": "Water for Drinking",
    "Sanitation": "Water for Sanitation",
    "Food Processing": "Water for Food Processing",
    "Textile Industry": "Water for Textile Industry",
    "Paper Industry": "Water for Paper Industry",
    "Chemical Industry": "Water for Chemical Industry",
    "Pharma Industry": "Water for Pharmaceutical Industry",
    "Construction": "Water for Construction",
    "Energy Production": "Water for Energy Production",
    "Oil Refining": "Water for Oil Refining",
    "Metals Production": "Water for Metals Production",
    "Auto Manufacturing": "Water for Automobile Manufacturing",
    "Electronics Manufacturing": "Water for Electronics Manufacturing",
    "Plastics Manufacturing": "Water for Plastics Manufacturing",
    "Leather Industry": "Water for Leather Industry",
    "Beverage Industry": "Water for Beverage Industry",
    "Pulp & Paper Industry": "Water for Pulp and Paper Industry",
    "Sugar Industry": "Water for Sugar Industry",
    "Cement Industry": "Water for Cement Industry",
    "Fertilizer Industry": "Water for Fertilizer Industry",
}

# Define categorical feature values
WATER_QUAL_NEEDS= {
    "Water Quality": ["Excellent",
                      "Good", 
                      "Fair", 
                      "Poor", 
                      "Very Poor",
                      "Toxic", 
                      "Polluted", 
                      "Eutrophic", 
                      "Saline",
                      "Acidic/Alkaline"
                      ],
    "Ethnicity": [
        "English", 
        "Mandarin Chinese", 
        "Spanish", 
        "French", 
        "Arabic", 
        "Hindi",
        "Bengali", 
        "Russian", 
        "Portuguese",
        "Japanese",
        "Swahili", 
        "Hausa",
        "Yoruba",
        "Zulu", 
        "Amharic",
        "Agni",
        "Baoule", 
        "Bron",
        "Asante"
        ],
    "Region": {
        "English": [
            "United States", 
            "United Kingdom",
            "Canada",
            "Australia", 
            "South Africa"
                    ],
        "Mandarin Chinese": [
            "China (Mainland China)",
            "Taiwan",
            "Singapore", 
            "Malaysia", 
            "Indonesia"
             ],
        "Spanish": [
            "Mexico",
            "United States (primarily in areas with a large Hispanic population)",
            "Spain", 
            "Colombia",
            "Argentina"
                    ],
        "French": [
            "France",
            "Democratic Republic of the Congo",
            "Canada (particularly in Quebec)", 
            "Belgium",
            "Cote d'Ivoire (Ivory Coast)"
                   ],
        "Arabic": [
            "Egypt", 
            "Saudi Arabia", 
            "Algeria", 
            "Morocco", 
            "Sudan"
                   ],
        "Hindi": [
            "India",
            "Nepal",
            "Fiji",
            "Trinidad and Tobago",
            "Guyana"
                  ],
        "Bengali": [
            "Bangladesh",
            "India (particularly in the state of West Bengal)",
            "West Bengal (India) is a major region."
            ],
        "Russian": [
            "Russia (primarily in the European part)",
            "Kazakhstan", 
            "Ukraine",
            "Belarus",
            "Kyrgyzstan"
                    ],
        "Portuguese": [
            "Brazil", 
            "Portugal",
            "Mozambique", 
            "Angola",
            "Guinea-Bissau"
                       ],
        "Japanese": [
            "Japan (natively spoken)",
            "Brazil (has a significant Japanese-speaking community)",
            "Hawaii, USA (also has a Japanese-speaking community)",
            "Peru (small Japanese-speaking community)",
            "Canada (particularly in Vancouver and Toronto)"
                     ],
        "Swahili": [
            "Kenya", 
            "Tanzania",
            "Uganda", 
            "Rwanda", 
            "Burundi",
            "Democratic Republic of Congo"
                    ],
        "Hausa": [
            "Nigeria","Niger"
                  ],
        "Yoruba": [
            "Nigeria","Benin", "Togo"
                   ],
        "Zulu": ["South Africa (particularly in the KwaZulu-Natal province)"
                 ],
        "Amharic": ["Ethiopia"],
        "Agni": ["Cote d'Ivoire"],
        "Baoule": ["Cote d'Ivoire"],
        "Bron": ["Cote d'Ivoire","Ghana"],
        "Asante": ["Ghana", "Cote d'Ivoire"],
        },
        # Random GDP per capita values
        # np.random.uniform(1000, 50000, num_samples).round(2),
    "Economic Status": [], # will define later    
}

# SDG6 Challenges dictionary with shorthand keys
SDG6_CHALLENGES = {
    "Lack of Access": "Access",
    "Water Scarcity": "Scarcity",
    "Water Pollution": "Pollution",
    "Ecosystem Degradation": "Ecosystems",
    "Governance Issues": "Governance",
}

ORE_TYPE = {
    'Type1': 'Gold Ore',
    'Type2': 'Iron Ore',
    'Type3': 'Copper Ore',
    'Type4': 'Silver Ore',
    'Type5': 'Lead Ore',
    'Type6': 'Zinc Ore',
    'Type7': 'Nickel Ore',
    'Type8': 'Tin Ore',
    'Type9': 'Bauxite',
    'Type10': 'Cobalt Ore',
    'Type11': 'Chromite',
    'Type12': 'Uranium Ore',
    'Type13': 'Manganese Ore',
    'Type14': 'Platinum Ore',
    'Type15': 'Tantalum Ore',
    'Type16': 'Vanadium Ore',
    'Type17': 'Molybdenum Ore',
    'Type18': 'Titanium Ore',
    'Type19': 'Lithium Ore',
    'Type20': 'Tungsten Ore',
    'Type21': 'Antimony Ore',
    'Type22': 'Mercury Ore',
    'Type23': 'Sulfur Ore',
    'Type24': 'Graphite Ore',
    'Type25': 'Diamond Ore',
    'Type26': 'Rare Earth Element Ores',
    'Type27': 'Phosphate Ore',
    'Type28': 'Gypsum Ore',
    'Type29': 'Fluorite Ore',
    'Type30': 'Barite Ore',
    'Type31': 'Asbestos Ore',
    'Type32': 'Boron Ore',
    'Type33': 'Potash Ore'
}

EXPLOSIVE_TYPE = {
    'Explosive1': 'ANFO (Ammonium Nitrate Fuel Oil)',
    'Explosive2': 'Water Gel Explosives',
    'Explosive3': 'Emulsion Explosives',
    'Explosive4': 'Dynamite',
    'Explosive5': 'Nitroglycerin',
    'Explosive6': 'Slurry Explosives',
    'Explosive7': 'Binary Explosives',
    'Explosive8': 'Boosters',
    'Explosive9': 'Detonating Cord',
    'Explosive10': 'C-4 (Plastic Explosive)',
    'Explosive11': 'Ammonium Nitrate',
    'Explosive12': 'Black Powder',
    'Explosive13': 'TNT (Trinitrotoluene)',
    'Explosive14': 'RDX (Cyclotrimethylenetrinitramine)',
    'Explosive15': 'PETN (Pentaerythritol Tetranitrate)',
    'Explosive16': 'ANFO Prills',
    'Explosive17': 'Cast Boosters',
    'Explosive18': 'Ammonium Nitrate Emulsion',
    'Explosive19': 'Nitrocellulose',
    'Explosive20': 'Aluminized Explosives',
    'Explosive21': 'Pentolite',
    'Explosive22': 'Semtex',
    'Explosive23': 'Nitroguanidine',
    'Explosive24': 'HMX (Cyclotetramethylenetetranitramine)',
    'Explosive25': 'Amatol',
    'Explosive26': 'Tetryl',
    'Explosive27': 'Composition B',
    'Explosive28': 'Water Gels with Sensitizers',
    'Explosive29': 'Nitrate Mixture Explosives',
    'Explosive30': 'Perchlorate Explosives',
    'Explosive31': 'Detonators (Non-Electric)',
    'Explosive32': 'Electric Detonators',
    'Explosive33': 'Electronic Detonators'
}

EQUIPMENT_TYPE = [
    'Excavator', 
    'Drill', 
    'Loader', 
    'Truck',
    "Articulated Haulers",
    "Asphalt Pavers",
    "Backhoe Loaders",
    "Blasthole Drills",
    "Bulldozers",
    "Cable Shovels",
    "Continuous Miners",
    "Conveyor Systems",
    "Crushing Equipment",
    "Draglines",
    "Drilling Rigs",
    "Dump Trucks",
    "Electric Rope Shovels",
    "Excavators",
    "Exploration Drills",
    "Feller Bunchers",
    "Forwarders",
    "Graders",
    "Harvesters",
    "Hydraulic Mining Shovels",
    "Jaw Crushers",
    "Loaders",
    "Material Handlers",
    "Milling Equipment",
    "Motor Graders",
    "Off-Highway Trucks",
    "Pipelayers",
    "Road Reclaimers",
    "Rock Drills",
    "Rotary Drills",
    "Scrapers",
    "Skid Steer Loaders",
    "Telehandlers",
    "Track Loaders",
    "Tracked Dozers",
    "Underground Mining Loaders",
    "Underground Mining Trucks",
    "Wheel Dozers",
    "Wheel Excavators",
    "Wheel Loaders",
    "Wheel Tractor-Scrapers",
    "Hydraulic Hammer",
    "Jumbos and Drifters",
    "Longwall Miners",
    "Roof Bolters",
    "Scooptrams",
    "Shotcrete Machines",
    "Shuttle Cars",
    "Stackers",
    "Reclaimers",
    "Screening Plants",
    "Haul Trucks",
    "Feeders",
    "Gyratory Crushers",
    "Cone Crushers",
    "Impact Crushers",
    "Hammer Mills",
    "Sizers"
]


COMMON_CROPS = [
    "Wheat", 
    "Rice",
    "Corn",
    "Barley", 
    "Soybeans",
    "Oats",
    "Rye",
    "Millet",
    "Sorghum",
    "Canola",
    "Sunflower",
    "Cotton",
    "Sugar Cane", 
    "Sugar Beet",
    "Potatoes",
    "Tomatoes",
    "Onions", 
    "Cabbage",
    "Carrots",
    "Lettuce",
    "Spinach", 
    "Broccoli", 
    "Garlic",
    "Cucumbers", 
    "Pumpkins",
    "Peppers",
    "Eggplants",
    "Zucchini", 
    "Squash",
    "Peas",
    "Green Beans",
    "Lentils", 
    "Chickpeas",
    "Almonds", 
    "Walnuts",
    "Peanuts", 
    "Cashews", 
    "Pistachios", 
    "Apples",
    "Oranges",
    "Bananas", 
    "Grapes",
    "Strawberries",
    "Blueberries",
    "Raspberries",
    "Blackberries",
    "Cherries", 
    "Peaches",
    "Pears", 
    "Plums",
    "Pineapples",
    "Mangoes",
    "Avocados"
]

COMMON_PESTICIDES = [
    'Herbicide',
    'Insecticide',
    'Fungicide'
    "Glyphosate", 
    "Atrazine",
    "2,4-Dichlorophenoxyacetic acid (2,4-D)", 
    "Dicamba",
    "Paraquat", 
    "Chlorpyrifos",
    "Metolachlor",
    "Imidacloprid",
    "Thiamethoxam", 
    "Clothianidin", 
    "Acetamiprid", 
    "Fipronil",
    "Bacillus thuringiensis (Bt)",
    "Neonicotinoids",
    "Pyrethroids",
    "Carbamates",
    "Organophosphates",
    "Sulfonylureas",
    "Roundup",
    "Liberty",
    "Malathion",
    "Diazinon", 
    "DDT", 
    "Methoxychlor",
    "Aldrin", 
    "Dieldrin", 
    "Endrin",
    "Chlordane",
    "Heptachlor", 
    "Hexachlorobenzene",
    "Mirex", 
    "Toxaphene",
    "Captan", 
    "Chlorothalonil", 
    "Mancozeb",
    "Maneb", 
    "Zineb",
    "Copper Sulphate",
    "Streptomycin",
    "Tetracycline", 
    "Difenoconazole", 
    "Propiconazole",
    "Cyproconazole", 
    "Azoxystrobin",
    "Chlorantraniliprole",
    "Abamectin", 
    "Spinosad", 
    "Bifenthrin",
    "Cyfluthrin", 
    "Deltamethrin", 
    "Permethrin", 
    "Cypermethrin",
    "Metam Sodium",
    "Methyl Bromide",
    "Chloropicrin",
    "Vapam"
]
AFRICAN_COUNTRIES= [
    "Algeria",
    "Angola",
    "Benin", 
    "Botswana",
    "Burkina Faso",
    "Burundi",
    "Cabo Verde", 
    "Cameroon",
    "Central African Republic", 
    "Chad",
    "Comoros", 
    "Congo",
    "Congo Democratic Republic", 
    "Cote d'Ivoire",
    "Djibouti",
    "Egypt", 
    "Equatorial Guinea", 
    "Eritrea", 
    "Eswatini",
    "Ethiopia",
    "Gabon",
    "Gambia", 
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Kenya", 
    "Lesotho",
    "Liberia",
    "Libya",
    "Madagascar",
    "Malawi", 
    "Mali",
    "Mauritania",
    "Mauritius",
    "Morocco", 
    "Mozambique",
    "Namibia",
    "Niger",
    "Nigeria",
    "Rwanda", 
    "Sao Tome and Principe",
    "Senegal", 
    "Seychelles",
    "Sierra Leone",
    "Somalia",
    "South Africa",
    "South Sudan",
    "Sudan", 
    "Tanzania",
    "Togo", 
    "Tunisia", 
    "Uganda",
    "Zambia",
    "Zimbabwe"
]
DIAGNOSIS_UNITS = {
    'age': 'years',
    'gender': 'category',
    'ethnicity': 'category',
    'weight': 'kg',
    'height': 'cm',
    'systolic': 'mmHg',
    'diastolic': 'mmHg',
    'heart_rate': 'beats/minute',
    'temperature': '°C',
    'blood_sugar': 'mg/dL',
    'cholesterol': 'mg/dL',
    'hemoglobin': 'g/dL',
    'history_of_diabetes': 'binary',
    'history_of_hypertension': 'binary',
    'history_of_heart_disease': 'binary',
    'respiratory_rate': 'breaths/minute',
    'oxygen_saturation': '%',
    'pain_score': 'scale (0 to 10)',
    'alt_levels': 'U/L',
    'creatinine_levels': 'mg/dL',
    'wbc_count': 'x10^3/uL',
    'bmi': 'kg/m^2',
    'daily_caloric_intake': 'calories',
    'dietary_restrictions': 'binary',
    'physical_activity_level': 'category',
    'smoking_status': 'binary',
    'alcohol_consumption': 'binary',
    'stress_level': 'scale (0 to 10)',
    'sleep_hours_per_night': 'hours',
    'mental_health_status': 'binary',
    'history_of_chronic_diseases': 'binary',
    'number_of_surgeries': 'count',
    'family_history_of_major_diseases': 'binary',
    'number_of_current_medications': 'count',
    'allergy_flags': 'binary',
    'employment_status': 'binary',
    'living_situation': 'category',
    'access_to_healthcare': 'binary',
    'flu_vaccine': 'binary',
    'covid_19_vaccine': 'binary',
    'other_vaccines': 'binary'
}

# Hydrogeological parameters with their definitions and importance in 
# assessing groundwater resources, mining operations, and environmental 
# management.
HYDRO_PARAMS = {
    "porosity": (
        "Measure of the void spaces in a material, indicating how much water "
        "a rock formation can store. Porosity is expressed as a percentage, "
        "reflecting the volume of voids within the material compared to its "
        "total volume."
    ),
    "permeability": (
        "Indicates how easily water can flow through rock formations, measured "
        "in Darcy (D) or millidarcy (mD). High permeability suggests that water "
        "can move freely through the rock, while low permeability indicates "
        "restricted water flow."
    ),
    "hydraulic_conductivity": (
        "Measures the ability of a formation to transmit water under a "
        "hydraulic gradient, typically expressed in meters per day (m/d). "
        "This parameter is essential for understanding the movement of water "
        "through aquifers and the potential for dewatering in mining operations."
    ),
    "transmissivity": (
        "Measure of how much water can be transmitted horizontally through an "
        "aquifer layer, expressed in square meters per day (m2/day). It integrates "
        "the hydraulic conductivity over the thickness of the aquifer, providing "
        "an overall estimate of the aquifer's capacity to convey water."
    ),
    "storativity": (
        "Also known as the storage coefficient, storativity reflects the volume "
        "of water an aquifer can release from storage per unit decline in hydraulic "
        "head, dimensionless for confined aquifers and typically very low."
    ),
    "specific_yield": (
        "The ratio of the volume of water that drains from a material due to "
        "gravity to the total volume of the material, expressed as a percentage. "
        "Specific yield is a critical parameter for evaluating water availability "
        "in unconfined aquifers."
    ),
    "fracture_density_and_orientation": (
        "Describes the density and orientation of fractures within rock formations, "
        "which are crucial for predicting water flow patterns and managing the risks "
        "associated with water or gas ingress in mining and hydrogeological studies."
    ),
    "water_table_depth": (
        "Direct measurement of the depth to the water table from the surface, "
        "expressed in meters (m). Understanding the water table depth is essential "
        "for groundwater exploration, well drilling, and assessing the potential "
        "for water inflow into mines."
    ),
    "aquifer_pressure": (
        "The pressure within an aquifer, which can be affected by geological "
        "stresses and is crucial for understanding water storage and movement. "
        "Aquifer pressure is measured in Pascals (Pa) or bars and influences the "
        "risk of water inflow into mining operations."
    ),
    "water_quality_parameters": (
        "Includes critical factors such as salinity, pH, and the presence of "
        "contaminants. These parameters determine the groundwater's suitability "
        "for various uses, including drinking, agriculture, and industrial processes, "
        "and are vital for environmental impact assessments."
    ),
    "temperature_gradients": (
        "Represents the variation in temperature with depth, indicating geothermal "
        "gradients. Temperature gradients are essential for understanding water "
        "quality, geothermal energy potential, and designing effective mine "
        "ventilation and cooling systems."
    )
}

# hydrogeological parameters crucial for deep mining with their roles
RELEVANT_HYDRO_PARAMS = {
    "permeability": (
        "Indicates risk of water ingress by showing how easily water can "
        "flow through rock formations. Essential for assessing the risk of "
        "water ingress into mining operations."
    ),
    "hydraulic_conductivity": (
        "Measures the rock's ability to transmit water under a hydraulic "
        "gradient. This parameter is crucial for understanding water movement "
        "and managing the risk of water ingress."
    ),
    "fracture_density_and_orientation": (
        "Density and orientation of fractures are crucial for planning mine "
        "layouts to avoid water or gas ingress hazards."
    ),
    "water_table_depth": ( 
        "The depth to the water table helps predict the potential for water "
        "inflow and the need for dewatering operations."
    ),
    "storativity": ( 
        "Storativity (or the storage coefficient) reflects the volume of "
        "water an aquifer can release from storage. Important for managing "
        "water control measures in confined aquifers."
    ),
    "specific_yield": ( 
        "Specific yield is the ratio of the volume of water that drains from "
        "a material due to gravity to the total volume of the material. "
        "Critical for evaluating water availability in unconfined aquifers."
    ),
    "aquifer_pressure": ( 
        "Aquifer pressure, influenced by geological stresses, affects the "
        "risk of water inflow from high-pressure aquifers, requiring careful "
        "management in mining operations."
    ),
    "temperature_gradients": ( 
        "Temperature gradients within the earth's crust can affect working "
        "conditions and mine stability. This parameter is important for "
        "the design of ventilation and cooling systems."
    ),
    "water_quality_parameters":(
        "Parameters such as salinity, pH, and the presence of contaminants "
        "are important for environmental impact management and ensuring the "
        "safety of discharged water from mining operations."
    )
}

HYDRO_PARAM_UNITS={
    "porosity": "% (percentage)",
    "permeability": "Darcy (D) or millidarcy (mD)",
    "hydraulic_conductivity": "m/D (meters per darcies)",
    "transmissivity": "m2/day (square meters per day)",
    "storativity": "dimensionless (volume of water per volume of aquifer)",
    "specific_yield": "% (percentage)",
    "water_table_depth": "m (meters)",
    "aquifer_pressure": "Pa (Pascal) or bar",
    "temperature": "°C (degrees Celsius)"
}

HYDRO_PARAM_RANGES = {
    "porosity": (
        0.01, 0.35, 
        # Porosity indicates the fraction of void space within a rock formation, 
        # expressed as a percentage. It plays a crucial role in determining 
        # the rock's ability to store water. Typical values range from 1% to 35%, 
        # varying widely across different geological formations.
    ),
    "permeability": (
        1e-5, 1e3, 
        # Permeability measures the ease with which fluids can flow through 
        # rock formations, quantified in Darcies. This parameter spans a broad 
        # spectrum, from less than one millidarcy in tight formations to 
        # thousands of Darcies in highly permeable sands or gravels.
    ),
    "hydraulic_conductivity": (
        1e-6, 1e-1, 
        # Hydraulic conductivity quantifies the capacity of a rock formation to 
        # transmit water under a hydraulic gradient, with units of meters per second (m/s). 
        # Values range from very low for materials like clay to moderately high for 
        # gravelly soils, reflecting the diverse hydrogeological characteristics.
    ),
    "transmissivity": (
        1e-4, 1e3, 
        # Transmissivity represents the ability of an aquifer to transmit water 
        # horizontally, measured in square meters per day (m2/day). It integrates 
        # hydraulic conductivity across the thickness of the aquifer, covering 
        # a wide range from low productivity to highly productive aquifers.
    ),
    "storativity": (
        1e-5, 0.2, 
        # Storativity, or the storage coefficient, describes the amount of water 
        # an aquifer releases or absorbs per unit area per unit change in head. 
        # For confined aquifers, this value is typically very low, whereas for 
        # unconfined aquifers, it can be as high as 20%.
    ),
    "specific_yield": (
        0.01, 0.3, 
        # Specific yield refers to the proportion of water that can be drained 
        # from the material due to gravity, applicable to unconfined aquifers. 
        # It is similar to porosity but only considers the water that can be 
        # freely mobilized and used.
    ),
    "water_table_depth": (
        1, 500, 
        # The depth to the water table, measured in meters, indicates the distance 
        # from the ground surface to the upper boundary of the groundwater. This 
        # depth can significantly vary depending on the geographical and 
        # hydrological conditions.
    ),
    "aquifer_pressure": (
        1e5, 1e7, 
        # Aquifer pressure, measured in Pascals, influences the movement and 
        # storage of groundwater. Values can range from relatively low pressures 
        # in shallow aquifers to very high pressures in deeper geological 
        # formations.
    ),
    "temperature": (
        10, 60, 
        # Groundwater temperature, measured in degrees Celsius, is affected by 
        # geothermal gradients and surface conditions. This parameter is essential 
        # for understanding thermal dynamics in aquifers and designing 
        # geothermal energy systems.
    )
}

# Note: The list and details are illustrative and based on generalized data; 
# specific figures andBased on the insights gathered from various sources,
# including the U.S. Geological Survey's Mineral Commodity Summaries 2023,
# Wikipedia's list of countries by mineral production, and Yahoo Finance's 
# including 68 countries known for their significant mineral production volumes,
# reserves, production capacities, and exports. This comprehensive list aims
# to highlight a diverse range of countries and their contributions to the 
# global mining sector, focusing on various minerals and their economic impact.

MINERAL_PROD_BY_COUNTRY = {
    "Australia": [
        "Vast reserves of bauxite, [iron ore, lithium]",
        "High production capacity for iron ore, lithium", 
        "Major exporter of lithium, iron ore", 
        ],
    "China": [
        "Large reserves of [coal, rare earth elements]",
        "World's top producer of several minerals including rare earths, coal",
        "Significant exporter of rare earth elements, coal"
        ],
    "Russia": [
        "Significant reserves of [palladium, nickel]",
        "Leading producer of palladium, nickel",
        "Major exporter of palladium, nickel"
        ],
    "United States": [
        "Large reserves of [gold, copper]", 
        "Top producer of gypsum, and significant production of copper, gold",
        "Major exporter of gypsum, significant exporter of copper, gold"
        ],
    "Canada": [
        "Substantial reserves of [potash, uranium]",
        "Leading producer of potash, uranium", 
         "Key exporter of potash, uranium"
         ],
    "Brazil": [
        "Rich in [iron ore, niobium]", 
        "Top producer of iron ore, niobium",
        "Leading exporter of iron ore, niobium"
        ],
    "South Africa": [
        "Huge reserves of [platinum, chromium]",
        "World's top producer of platinum, chromium",
        "Major exporter of platinum, chromium"
        ],
    "India": [
        "Significant reserves of [coal, iron ore]",
        "Major producer of coal, iron ore",
        "Substantial exporter of iron ore"
        ],
    "Indonesia": [
        "Large coal reserves, significant [gold, nickel]", 
        "Top coal exporter, major producer of gold, nickel",
        "World's top coal exporter, significant exporter of gold, nickel"
        ],
    "Chile": [
        "World's largest [copper] reserves", 
        "Top producer and exporter of copper", 
        "Leading exporter of copper"
        ],
    "Peru": [
        "Significant [silver, copper], gold reserves", 
        "Major producer of silver, copper", 
        "Important exporter of silver, copper"
        ],
    "Kazakhstan": [
        "Large reserves of [uranium, chromium]",
        "Top producer of uranium, significant producer of chromium", 
        "Leading exporter of uranium"
        ],
    "Argentina": [
        "Rich in [lithium, silver], copper", 
        "Growing producer of lithium",
        "Emerging exporter of lithium and silver"
        ],
    "Philippines": [
        "Significant [nickel], gold reserves",
        "Major producer of nickel",
        "Top exporter of nickel"
        ],
    "Ghana": [
        "Major [gold] producing country", 
        "Significant gold production capacity", 
        "Important gold exporter"
        ],
    "Mexico": [
        "World's largest [silver] reserves",
        "Top producer of silver", 
        "Leading exporter of silver"
        ],
    "Sweden": [
        "Significant [iron ore, copper] reserves",
        "Major producer of iron ore", 
        "Key exporter of iron ore, copper"
        ],
    "Zambia": [
        "Large [copper] reserves", 
        "Second-largest copper producer in Africa",
        "Significant copper exporter"
        ],
    "Democratic Republic of Congo": [
        "World's largest [cobalt] reserves",
        "Top producer of cobalt", 
        "Major exporter of cobalt"
        ],
    "Zimbabwe": [
        "Significant [platinum, diamond]",
        "Major producer of platinum",
        "Important exporter of platinum"
        ],
    "Mongolia": [
        "Rich in [coal], copper, gold",
        "Significant coal and copper production", 
        "Important exporter of coal"
        ],
    "Saudi Arabia": [
        "Large reserves of [phosphate], gold", 
        "Major producer of phosphate", 
        "Key exporter of phosphate"
        ],
    "United Arab Emirates": [
        "Significant [aluminum] producer", 
        "Major aluminum production capacity", 
        "Leading exporter of aluminum"
        ],
    "Turkey": [
        "Substantial [marble, boron] reserves",
        "Top producer of boron", 
        "Major exporter of marble, boron"
        ],
    "Norway": [
        "Significant producer of petroleum, [metals]",
        "Major oil exporter, significant metals production", 
        "Key exporter of metals, oil"
        ],
    "Vietnam": [
        "Rich in [bauxite], rare earth elements",
        "Growing bauxite producer",
        "Emerging exporter of bauxite"
        ],
    "Nigeria": [
        "Significant [oil], gas reserves; emerging in minerals",
        "Major oil producer, growing in minerals like gold",
        "Top oil exporter, emerging mineral exporter"
        ],
    "Tanzania": [
        "Rich in [gold, gemstones], diamond",
        "Major gold producer",
        "Important gold, gemstone exporter"
        ],
    "Papua New Guinea": [
        "Significant [gold, copper] reserves",
        "Major gold, copper producer", 
        "Key exporter of gold, copper"
        ],
    "Iran": [
        "Large reserves of copper, iron ore, [zinc]", 
        "Significant producer of copper, iron ore", 
        "Important exporter of minerals"
        ],
    "Ukraine": [
        "Rich in [iron ore], coal", 
        "Significant production of iron ore, coal",
        "Important exporter of iron ore"
        ],
    "Poland": [
        "Substantial [coal] reserves",
        "Major coal producer",
        "Key coal exporter"
        ],
    "Bolivia": [
        "Large deposits of [silver], lithium",
        "Major producer of silver, growing lithium production",
        "Significant exporter of silver"
        ],
    "Namibia": [
        "Significant [uranium], diamond",
        "Major uranium producer",
        "Important exporter of uranium, diamond"
        ],
    "Botswana": [
        "World-leading [diamond] reserves",
        "Top diamond producer", 
        "Major diamond exporter"
        ],
    "New Zealand": [
        "Considerable [coal, gold] reserves",
        "Significant coal and gold production", 
        "Exporter of coal and gold"
        ],
    "Finland": [
        "Rich in [nickel, chromium]",
        "Significant producer of nickel", 
        "Key exporter of nickel, chromium"
        ],
    "Mali": [
        "Significant [gold] reserves",
        "Major gold producer",
        "Important gold exporter"
        ],
    "Burkina Faso": [
        "Growing [gold] production", 
        "Significant gold producer", 
        "Emerging gold exporter"
        ],
    "Colombia": [
        "Large [coal, emerald] reserves",
        "Major coal producer and top emerald producer", 
        "Key exporter of coal and emeralds"
        ],
    "Qatar": [
        "Rich in petroleum, [natural gas]", 
        "Major petroleum and natural gas producer",
        "Top exporter of liquefied natural gas"
        ],
    "Egypt": [
        "Substantial gold, [phosphate]", 
        "Growing gold producer, major phosphate producer",
        "Important exporter of phosphate"
        ],
    "Oman": [
        "Significant [gypsum], copper",
        "Major gypsum producer",
        "Leading exporter of gypsum"
        ],
    "Angola": [
        "Rich in [diamond, oil]", 
        "Major diamond producer, significant oil production",
        "Key exporter of diamond and oil"
        ],
    "Kuwait": [
        "Large [petroleum] reserves", 
        "Significant petroleum production", 
        "Major petroleum exporter"
        ],
    "Libya": [
        "Substantial [oil] reserves",
        "Major oil producer",
        "Significant oil exporter"
        ],
    "Bahrain": [
        "Significant [oil, natural gas]", 
        "Major oil and natural gas production",
        "Key exporter of petroleum products"
        ],
    "Bangladesh": [
        "Considerable [natural gas]", 
        "Significant natural gas production", 
        "Emerging exporter of natural gas"
        ],
    "Cuba": [
        "Rich in [nickel, cobalt]",
        "Major nickel producer", 
        "Important exporter of nickel and cobalt"
        ],
    "Venezuela": [
        "Large petroleum, [oil] reserves",
        "Major oil producer", 
        "Key oil exporter"
        ],
    "Suriname": [
        "Significant [bauxite] reserves",
        "Notable bauxite production",
        "Bauxite exporter"
        ],
    "Guinea": [
        "World-leading [bauxite] reserves",
        "Top bauxite producer", 
        "Major bauxite exporter"
        ],
    "Senegal": [
        "Significant [phosphate] reserves",
        "Phosphate production", 
        "Phosphate exporter"
        ],
    "Cameroon": [
        "Emerging [iron ore, bauxite] production",
        "Developing mining sector",
        "Potential exporter of iron ore and bauxite"
        ],
    "Sierra Leone": [
        "Rich in [diamond]", 
        "Significant diamond producer",
        "Diamond exporter"
        ],
    "Cote d'Ivoire": [
        "Growing [gold, manganese] production",
        "Emerging in gold and manganese", 
        "Potential gold and manganese exporter"
        ],
    "Liberia": [
        "Significant [iron ore] reserves",
        "Iron ore production",
        "Iron ore exporter"
        ],
    "Mozambique": [
        "Significant coal, [titanium] reserves",
        "Coal and titanium production", 
        "Coal and titanium exporter"
        ],
    "Madagascar": [
        "Large [graphite, nickel] reserves",
        "Graphite and nickel production", 
        "Graphite and nickel exporter"
        ],
    "Lesotho": [
        "Significant [diamond] reserves",
        "Diamond production",
        "Diamond exporter"
        ],
    "Ethiopia": [
        "Emerging in [opal, gold, tantalum]",
        "Opal, gold, and tantalum production",
        "Opal, gold, and tantalum exporter"
        ],
    "Kyrgyzstan": [
        "large [gold, coal, antimony] reserves", 
        "Gold, coal, and antimony production",
        "Gold, coal, and antimony exporter"
        ],
    "Tajikistan": [
        "Significant [silver, gold] reserves",
        "Silver and gold production",
        "Silver and gold exporter"
        ],
    "Myanmar": [
        "Rich in [jade, gems, tin]",
        "Jade, gems, and tin production", 
        "Jade, gems, and tin exporter"
        ],
    "Laos": [
        "Emerging [potash, gold] production",
        "Developing mining sector for potash and gold", 
        "Potential potash and gold exporter"
        ],
    "Brunei": [
        "Large [petroleum, natural gas] reserves",
        "Petroleum and natural gas production",
        "Petroleum and natural gas exporter"
        ],
    "Turkmenistan": [
        "Significant [natural gas] reserves",
        "Natural gas production", 
        "Natural gas exporter"
        ],
    "Uzbekistan": [
        "Large [gold, uranium, copper] reserves", 
        "Gold, uranium, and copper production",
        "Gold, uranium, and copper exporter"
        ]
}

RESERVES_DESCR = {
    'estimate_reserves': [
        'Estimated total amount of mineral resources available for extraction, considering economic viability',
        'tons or metric tons'
    ], 
    'estimated_production': [
        'Estimated amount of mineral produced annually',
        'tons or metric tons'
    ],
    'extraction_cost': [
        'Estimate of the cost associated with extracting a unit of the mineral', 
        'USD per ton'
    ],
    'grade': [
        'Concentration of mineral in the ore body', 
        '% or g/ton'
    ],
    'depth': [
        'Depth at which the mineral deposit is found', 
        'meters'
    ],
    'accessibility': [
        'Ease of access to the mineral deposit considering terrain and infrastructure', 
        'qualitative (e.g., high, medium, low)'
    ],
    'environmental_impact': [
        'Assessment of the environmental footprint of extracting the mineral', 
        'qualitative or quantitative (e.g., CO2 emissions)'
    ],
    'reserve_life': [
        'Estimated duration the mineral deposit is expected to last at current extraction rates', 
        'years'
    ],
    'ownership': [
        'Entity that holds the rights to extract the mineral', 
        'text (company name or government)'
    ],
    'regulatory_status': [
        'Legal or regulatory framework governing mineral extraction in the location', 
        'text (e.g., permitted, pending approval)'
    ],
    'market_demand': [
        'Current global demand for the mineral', 
        'qualitative (e.g., high, medium, low)'
    ],
    'historical_data': [
        'Past production volumes, historical prices, or exploration data', 
        'varies (e.g., tons, USD)'
    ],
    'associated_minerals': [
        'Other minerals present in the deposit that could provide economic opportunities', 
        'list of strings'
    ],
    'recovery_rate': [
        'Percentage of mineral successfully extracted and processed from the ore', 
        '%'
    ],
    'technology_used': [
        'Mining and processing technologies applied in extraction', 
        'text (e.g., heap leaching, flotation)'
    ]
}


COUNTRY_REGION = {
 'Australia': 'Oceania',
 'China': 'Asia',
 'Russia': 'Europe/Asia',
 'United States': 'North America',
 'Canada': 'North America',
 'Brazil': 'South America',
 'South Africa': 'Africa',
 'India': 'Asia',
 'Indonesia': 'Asia',
 'Chile': 'South America',
 'Peru': 'South America',
 'Kazakhstan': 'Europe/Asia',
 'Argentina': 'South America',
 'Philippines': 'Asia',
 'Ghana': 'Africa',
 'Mexico': 'North America',
 'Sweden': 'Europe',
 'Zambia': 'Africa',
 'Democratic Republic of Congo': 'Africa',
 'Zimbabwe': 'Africa',
 'Mongolia': 'Asia',
 'Saudi Arabia': 'Asia',
 'United Arab Emirates': 'Asia',
 'Turkey': 'Europe/Asia',
 'Norway': 'Europe',
 'Vietnam': 'Asia',
 'Nigeria': 'Africa',
 'Tanzania': 'Africa',
 'Papua New Guinea': 'Oceania',
 'Iran': 'Asia',
 'Ukraine': 'Europe',
 'Poland': 'Europe',
 'Bolivia': 'South America',
 'Namibia': 'Africa',
 'Botswana': 'Africa',
 'New Zealand': 'Oceania',
 'Finland': 'Europe',
 'Mali': 'Africa',
 'Burkina Faso': 'Africa',
 'Colombia': 'South America',
 'Qatar': 'Asia',
 'Egypt': 'Africa',
 'Oman': 'Asia',
 'Angola': 'Africa',
 'Kuwait': 'Asia',
 'Libya': 'Africa',
 'Bahrain': 'Asia',
 'Bangladesh': 'Asia',
 'Cuba': 'North America',
 'Venezuela': 'South America',
 'Suriname': 'South America',
 'Guinea': 'Africa',
 'Senegal': 'Africa',
 'Cameroon': 'Africa',
 'Sierra Leone': 'Africa',
 "Cote d'Ivoire": 'Africa',
 'Liberia': 'Africa',
 'Mozambique': 'Africa',
 'Madagascar': 'Africa',
 'Lesotho': 'Africa',
 'Ethiopia': 'Africa',
 'Kyrgyzstan': 'Asia',
 'Tajikistan': 'Asia',
 'Myanmar': 'Asia',
 'Laos': 'Asia',
 'Brunei': 'Asia',
 'Turkmenistan': 'Asia',
 'Uzbekistan': 'Asia'
}

WATER_RESERVES_LOC = [
    "Emerald City", "Silver Lake", "Mystic River", "Crescent Reservoir", "Twin Peaks",
    "Golden Valley", "Sapphire Bay", "Echo Lake", "Willow Creek", "Ravenwood Forest",
    "Crystal Springs", "Marble City", "Iron Mountain", "Velvet River", "Azure Dam",
    "Falcon Heights", "Garnet Town", "Diamond Desert", "Amber Plains", "Obsidian Island",
    "Jade Harbor", "Quartz Quarry", "Coral Cove", "Opal Ocean", "Moonlight City",
    "Starfall River", "Sunrise Reservoir", "Twilight Town", "Midnight Lake", "Dawn Valley",
    "Dusk Dam", "Aurora Bay", "Solstice Spring", "Equinox Estate", "Zenith Peak",
    "Nadir City", "Phoenix River", "Serenity Lake", "Arcadia", "Utopia",
    "Atlantis", "El Dorado", "Avalon", "Shangri-La", "Camelot",
    "Brigadoon", "Olympus", "Eden", "Nirvana", "Elysium",
    "Babylon", "Zion", "Arcane River", "Mystic Marsh", "Celestial City",
    "Pegasus Plains", "Dragon's Den", "Griffin Grove", "Hydra Lake", "Chimera Creek",
    "Mermaid Lagoon", "Centaur Field", "Titan Town", "Valkyrie Valley", "Oracle Ocean",
    "Minotaur Mountain", "Siren Sea", "Cyclops City", "Elfwood", "Goblin Gorge",
    "Hobbiton Hills", "Dwarf Dam", "Vampire Village", "Werewolf Woods", "Zombie Zone",
    "Ghost Town", "Witchwater", "Wizard's Way", "Fairy Forest", "Demon's Desert",
    "Angel's Aerie", "Necropolis", "Spartan Springs", "Viking Village", "Samurai Sanctuary",
    "Pirate's Port", "Knight's Keep", "Galaxy Garden", "Meteor Meadow", "Comet Cove",
    "Planet Plaza", "Starlight Strand", "Moonbeam Marina", "Solaris", "Cosmos City",
    "Nebula Nook", "Quantum Quay", "Interstellar Inlet", "Galactic Gate", "Celestia",
    "Ruby Ridge", "Sapphire Shore", "Emerald Estuary", "Diamond Dale", "Amethyst Arbor",
    "Turquoise Trail", "Opal Outpost", "Pearl Port", "Quartz Quay", "Topaz Terrace",
    "Garnet Glen", "Jade Junction", "Lapis Lane", "Moonstone Meadow", "Sunstone Square",
    
    "Aquamarine Alley", "Citrine Court", "Peridot Park", "Zircon Zephyr", "Onyx Oasis",
    "Beryl Boulevard", "Malachite Mount", "Agate Avenue", "Rhodolite Road", "Spinel Street",
    "Tanzanite Town", "Tourmaline Trail", "Iolite Island", "Kunzite Key", "Larimar Land",
    "Morganite Mountain", "Heliodor Harbor", "Sphene Springs", "Zoisite Zone", "Chalcedony City",
    "Moon Harbour", "Starlight Bay", "Comet Corner", "Asteroid Arch", "Meteorite Meadow",
    "Galaxy Gateway", "Nebular Nook", "Stellar Stream", "Cosmic Cove", "Aurora Alley",
    "Eclipse Estate", "Solaris Station", "Planetoid Plaza", "Meteor Marsh", "Black Hole Basin",
    "Quasar Quarter", "Pulsar Path", "Supernova Sector", "Event Horizon Estate", "Wormhole Way",
    "Lightyear Lane", "Gravity Grove", "Orbit Orchard", "Cosmos Canyon", "Astro Arch",
    "Nova Neighborhood", "Lunar Lake", "Satellite Street", "Rocket Ridge", "Milky Way Meadow",
    "Interplanetary Park", "Exoplanet Estate", "Solar Flare Square", "Magnetar Mount", 
    "Neutron Star Nook", "Gamma Ray Garden", "Spaceport Spire", "Alien Alley", "UFO Upland",
    "Extraterrestrial Terrace","Singularity Street", "Dimensional Drive", "Time Traveler's Trail",
    "Multiverse Manor","Quantum Quarters", "Parallel Universe Path", "Void Valley", 
    "Cosmic Cluster Court", "Galactic Gateway", "Starship Street","Astrophysics Avenue",
    "Celestial Circuit", "Orbital Outpost", "Space Station Square", "Astronaut's Aisle",
    "Rocket Range", "Intergalactic Inn", "Mars Meadow", "Venus Valley", "Jupiter Junction",
    "Saturn Street", "Mercury Mount", "Neptune Nook", "Uranus Upland", "Pluto Path", 

    "Nile Vista", "Sahara Solitude", "Kilimanjaro Peak", "Serengeti Plains", "Zanzibar Shore",
    "Timbuktu Trails", "Cairo Crossroads", "Marrakech Market", "Okavango Oasis", "Victoria Falls View",
    "Atlas Ascent", "Limpopo Lane", "Cape Town Corner", "Kalahari Keep", "Madagascar Haven",
    "Giza Gateway", "Luxor Landing", "Aswan Acre", "Djibouti Dunes", "Mombasa Marina",
    "Namib Nook", "Tangier Terrace", "Freetown Freeway", "Lagos Lagoon", "Accra Avenue",
    "Dakar Drive", "Addis Ababa Alley", "Harare Heights", "Luanda Loop", "Maputo Mews",
    "Algiers Arch", "Benin Bay", "Casablanca Cove", "Dar Es Salaam Dock", "Entebbe Edge",
    "Fes Fountain", "Gaborone Grove", "Hargeisa Hill", "Ibadan Isle", "Jinja Junction",
    "Kampala Knoll", "Libreville Lane", "Maseru Meadow", "Nairobi Niche", "Ouagadougou Outpost",
    "Porto-Novo Path", "Quelimane Quay", "Rabat Ridge", "Sfax Stream", "Tripoli Trail",
    "Umtata Upland", "Vilankulo Valley", "Windhoek Way", "Xai-Xai Xanadu", "Yamoussoukro Yard",
    "Zinder Zone", "Abeokuta Avenue", "Bamako Bend", "Conakry Crescent", "Djenné Dome",
    "Eldoret Expanse", "Fianarantsoa Field", "Goma Gateway", "Hwange Horizon", "Inhambane Inlet",
    "Juba Junction", "Kisumu Knoll", "Luangwa Loop", "Mopti Mount", "Nouakchott Nook",
    "Oran Oasis", "Pemba Port", "Qunu Quarry", "Rustenburg Ridge", "Sokoto Spring",
    "Toubkal Tower", "Ubombo Utopia", "Volta Vista", "Wadi Waterfall", "eXplore eXpanse",
    "Yankari Yard", "Zambezi Zenith", "Agadez Arch", "Bujumbura Beach", "Chobe Chateau",
    "Draa Drift", "Etosha Edge", "Fish River Fault", "Giraffe Grounds", "Hippo Haven",
    "Cote d'Ivoire Isle", "Jozini Jewel", "Kafue Key", "Lalibela Loft", "Mali Mesa",
    "Ngorongoro Niche", "Omo Outback", "Pilanesberg Peak", "Quagga Quarters", "Rovos Rail",
    "Swakopmund Sands", "Tsavo Terrace", "Uganda Uplift", "Virunga Valley", "Watamu Wave"
]

MINERAL_RESERVES_DETAILS= {
    "Australia": {
        "estimated_reserves": "[1 billion] tonnes of iron ore, 160,000 tonnes of lithium (2020)",
        "extraction_cost": "[Low] to medium, depending on the mineral and location",
        "grade": "[High]-grade iron ore (>60% Fe content), Lithium concentration varies",
        "accessibility": "[High], well-established mining infrastructure",
        "reserve_life": "[50+] years for major commodities",
        "ownership": "[Various], including BHP, Rio Tinto, and smaller mining companies",
        "regulatory_status": "Strict environmental and mining regulations, supportive [government policies]",
        "market_demand": "[High], especially in Asia for iron ore and globally for lithium",
        "historical_data": "[Decades of extensive mining history], particularly for iron, gold, and bauxite",
        "associated_minerals": "[Gold, bauxite, nickel], and many others",
        "technology_used": "[Open-pit mining], underground mining, and various processing technologies"
    },
    "China": {
        "estimated_reserves": "[3.7 billion] tonnes of coal (2020), dominant in rare earth elements",
        "extraction_cost": "Varies [wide]ly across provinces",
        "grade": "Varies; coal mines can be [high] or low energy content",
        "accessibility": "[Mixed]; vast country with some remote mining areas",
        "reserve_life": "[30+] years for coal, decades for rare earth elements",
        "ownership": "[State-owned enterprises] predominantly, with some private and foreign companies",
        "regulatory_status": "Increasingly strict environmental regulations, significant [government oversight]",
        "market_demand": "[High] internally and for exports, especially for rare earth elements",
        "historical_data": "[Long mining history], world's top coal producer for decades, major role in rare earths",
        "associated_minerals": "A wide range including [iron, aluminum, gold], and more",
        "technology_used": "A range from traditional to [high-tech], especially in rare earth element processing"
    },
    
    "Russia": {
        "estimated_reserves": "[7.1 billion], Large reserves of palladium, nickel",
        "extraction_cost": "[Varies], depending on the mineral and region",
        "grade": "[High]-grade palladium and nickel ores",
        "accessibility": "[Varied] due to vast and remote locations",
        "reserve_life": "[Several decades] for key minerals",
        "ownership": "[State-controlled] companies like Norilsk Nickel",
        "regulatory_status": "[Strict regulations] with significant government oversight",
        "market_demand": "[High] globally, especially for palladium used in automotive catalysts",
        "historical_data": "[Long history of mining], particularly in the Norilsk region for nickel",
        "associated_minerals": "[Platinum, copper, and other PGMs] often found with nickel and palladium",
        "technology_used": "[Advanced mining] and metallurgical processing"
    },
    "United States": {
        "estimated_reserves": "Significant reserves with [4.8 billion] of gold and copper",
        "extraction_cost": "Medium to [high], depending on the mining depth and ore grade",
        "grade": "Varied grades, with some [world-class gold] deposits in Nevada",
        "accessibility": "[Good], with developed infrastructure and technology",
        "reserve_life": "[Over 20 years] for major mines, with ongoing exploration",
        "ownership": "[Mixed ownership] including major companies like Freeport-McMoRan and Newmont",
        "regulatory_status": "[Comprehensive environmental] and mining regulations",
        "market_demand": "[Stable to high], with gold as a hedge against economic uncertainty",
        "historical_data": "[Rich mining history], dating back to the 19th-century gold rushes",
        "associated_minerals": "[Silver, lead, and zinc], commonly associated with copper and gold deposits",
        "technology_used": "[Heap leaching] for gold extraction, [open-pit] and [underground mining] for copper"
    },
    "Canada": {
        "estimated_reserves": "Major reserves of potash and uranium with estimated [4.8 billion], among the world's largest",
        "extraction_cost": "Low for potash, [medium] for uranium due to deeper deposits",
        "grade": "[High-grade uranium] deposits in Saskatchewan, world-leading potash grades",
        "accessibility": "[High] for potash, medium for uranium due to environmental considerations",
        "reserve_life": "[Extensive], with decades of production capacity",
        "ownership": "[Diverse], with companies like Cameco (uranium) and Nutrien (potash)",
        "regulatory_status": "[Strict], with a focus on environmental protection and safety",
        "market_demand": "Steady for potash (fertilizer ingredient), [growing for uranium] (nuclear energy)",
        "historical_data": "Decades of [sustainable mining practices], particularly in Saskatchewan",
        "associated_minerals": "[Sylvite and halite] in potash deposits, rare earth elements near uranium mines",
        "technology_used": "[Solution mining] for potash, conventional and [ISR] for uranium"
    },
    "Brazil": {
        "estimated_reserves": "[100 of millions of tonnes] of iron ore, [thousands of tonnes] of niobium",
        "extraction_cost": "Low for iron ore due to surface mining, [medium] for niobium",
        "grade": "[High]-grade iron ore deposits, significant grades of niobium",
        "accessibility": "[High] for major mining areas with well-developed infrastructure",
        "reserve_life": "[Over 100 years] for iron ore, [extensive] for niobium",
        "ownership": "[Mixed] with major players like Vale in iron ore",
        "regulatory_status": "Subject to [environmental licensing], with a push towards sustainable mining",
        "market_demand": "[High] globally for iron ore, especially from China; stable for niobium",
        "historical_data": "[Centuries of mining history], especially in the Minas Gerais region",
        "associated_minerals": "[Gold, manganese, and quartz], commonly found with iron ore",
        "technology_used": "[Open-pit mining] for iron ore, various extraction techniques for niobium"
    },
    "South Africa": {
        "estimated_reserves": "[1 Millions of ounces] of platinum annually, [hundreds of thousands of tonnes] of chromium",
        "extraction_cost": "[Medium to high], particularly for deep underground platinum mines",
        "grade": "[High]-grade PGM (Platinum Group Metals) deposits, high-grade chromite",
        "accessibility": "[Variable], with some of the deepest mines in the world",
        "reserve_life": "[Several decades] for both platinum and chromium",
        "ownership": "[Diverse], including Anglo American Platinum, Impala Platinum, and others",
        "regulatory_status": "Complex, with significant [regulatory oversight] on mining operations",
        "market_demand": "[High] for platinum in automotive catalysts and jewelry, stable for chromium",
        "historical_data": "Rich in [mining history], with platinum and chromium mining dating back decades",
        "associated_minerals": "[Vanadium, palladium, rhodium], and other PGMs",
        "technology_used": "[Underground mining] and [concentrating] for platinum, [open-pit] and [underground] for chromium"
    },
    "India": {
        "estimated_reserves": "[100 of millions of tonnes] of coal annually, [tens of millions of tonnes] of iron ore",
        "extraction_cost": "[Low to medium], depending on the mine location and depth",
        "grade": "Varied coal grades, from [high]-quality coking coal to lower-grade thermal coal; high-grade iron ore",
        "accessibility": "[Moderate], with challenges in some remote areas",
        "reserve_life": "[Extensive], particularly for coal",
        "ownership": "[State-owned and private] entities, including Coal India Ltd for coal",
        "regulatory_status": "[Evolving], with recent reforms to encourage foreign investment",
        "market_demand": "[High] domestic demand for coal in power generation; strong demand for iron ore",
        "historical_data": "Long-standing [mining tradition], with ancient mining activities dating back over 2,000 years",
        "associated_minerals": "[Bauxite, manganese, and limestone], often found in proximity to coal and iron ore deposits",
        "technology_used": "[Open-pit] and [underground mining] for coal; [surface and underground mining] for iron ore"
    },
    "Indonesia": {
        "estimated_reserves": "[100 of millions of tonnes] of coal annually, significant quantities of gold and nickel",
        "extraction_cost": "Low for coal due to open-pit mining, [variable] for gold and nickel",
        "grade": "High-grade [nickel] ore, variable gold ore grades",
        "accessibility": "[Good] for coal, with challenges in remote gold and nickel mining areas",
        "reserve_life": "[Decades] for coal, substantial for gold and nickel",
        "ownership": "[Mixed], with major Indonesian and international mining companies",
        "regulatory_status": "Complex, with significant [government oversight] and evolving regulations",
        "market_demand": "[High] for coal in Asia; growing for nickel due to electric vehicle battery production",
        "historical_data": "[Rich mining history], particularly in coal and tin mining",
        "associated_minerals": "Tin, bauxite, and others",
        "technology_used": "[Open-pit mining] for coal, various methods for gold and nickel"
    },
    "Chile": {
        "estimated_reserves": "[1 Millions of tonnes] of copper annually",
        "extraction_cost": "Medium, due to large-scale [open-pit] and underground mining",
        "grade": "[High]-grade copper mines, especially in the Norte Grande region",
        "accessibility": "[Excellent], with advanced mining infrastructure",
        "reserve_life": "[Over 50 years], with ongoing exploration extending reserves",
        "ownership": "[State-owned Codelco], along with international mining giants",
        "regulatory_status": "[Strict] environmental and water use regulations",
        "market_demand": "[Very high], as the backbone of global copper supply",
        "historical_data": "[Centuries of copper mining], dating back to pre-colonial times",
        "associated_minerals": "Molybdenum, gold, and silver",
        "technology_used": "Primarily [open-pit mining], with some of the world's largest copper mines"
    },
    "Peru": {
        "estimated_reserves": "Significant quantities at around [200 millions] of silver and copper, notable gold production",
        "extraction_cost": "[Variable], with some high-altitude mines increasing operational costs",
        "grade": "[High-grade] silver mines, substantial copper deposits",
        "accessibility": "Challenging due to Andean [mountainous terrain]",
        "reserve_life": "[Long-term prospects] with extensive mineral wealth",
        "ownership": "[Diverse], with significant international investment",
        "regulatory_status": "[Evolving], with community and environmental considerations",
        "market_demand": "High for [silver and copper], with global industrial and investment demand",
        "historical_data": "Ancient mining culture, [rich in precious metals], with modern development",
        "associated_minerals": "[Lead, zinc, and tin]",
        "technology_used": "[Underground and open-pit mining], adapting to challenging geography"
    },
    "Kazakhstan": {
        "estimated_reserves": "Leading global producer of [1.8 billion] uranium, substantial chromium production",
        "extraction_cost": "Low to medium for uranium, [higher for chromium]",
        "grade": "[Very high-grade] uranium deposits, significant chromium resources",
        "accessibility": "Good, with vast steppe regions offering [easier mining access]",
        "reserve_life": "[Extensive], especially for uranium with strategic reserves",
        "ownership": "[State-dominated], with national companies controlling key assets",
        "regulatory_status": "[Supportive], with strategic interest in uranium production",
        "market_demand": "High for uranium [for nuclear energy], stable for chromium [used in stainless steel]",
        "historical_data": "[Significant mining sector], with post-Soviet expansion",
        "associated_minerals": "Coal, iron, and rare earth elements",
        "technology_used": "[In-situ leaching] for uranium, [open-pit mining] for chromium"
    },
    "Argentina": {
        "estimated_reserves": "Rapidly increasing lithium production around [ 600 million], significant silver resources",
        "extraction_cost": "[Low] for lithium from salt flats, variable for silver",
        "grade": "High concentration of lithium in [salars], variable silver ore grades",
        "accessibility": "[Challenging], with remote and high-altitude mining areas",
        "reserve_life": "[Long-term potential] for lithium, significant for silver",
        "ownership": "[Mixed], with international companies involved in lithium projects",
        "regulatory_status": "Favorable [government policies] for lithium, evolving for other minerals",
        "market_demand": "[Surging] for lithium due to electric vehicle battery market, stable for silver",
        "historical_data": "Recent development in lithium, [long history] of silver mining",
        "associated_minerals": "Copper, gold, and boron",
        "technology_used": "[Brine extraction] for lithium, various for silver and copper"
    },
    "Philippines": {
        "estimated_reserves": "Major nickel producer, with [millions of tonnes] annually",
        "extraction_cost": "[Low to medium], depending on the mine",
        "grade": "High-grade [nickel] laterite ores",
        "accessibility": "[Varied], with several large-scale mining operations on remote islands",
        "reserve_life": "[Decades], given the extensive nickel laterite resources",
        "ownership": "[Mixed], with both local and international mining companies",
        "regulatory_status": "[Complex] regulatory environment with a focus on sustainable mining",
        "market_demand": "High for [nickel], used in stainless steel and batteries",
        "historical_data": "[Significant] nickel production history, with growth in recent years",
        "associated_minerals": "Chromite, copper, and gold",
        "technology_used": "[Surface mining], particularly strip mining for laterite nickel ores"
    },
    "Ghana": {
        "estimated_reserves": "[Millions of ounces] of gold annually",
        "extraction_cost": "Medium, due to [deep underground] and surface operations",
        "grade": "Varied, with some [high-grade] underground deposits",
        "accessibility": "[Good], with established mining sector and infrastructure",
        "reserve_life": "[Over 20 years], with ongoing exploration and development",
        "ownership": "[Diverse], including major global mining companies",
        "regulatory_status": "[Supportive] government policies with an emphasis on local community benefits",
        "market_demand": "[High] globally, with gold as a key investment and jewelry commodity",
        "historical_data": "[Rich] with a gold mining history dating back to the 15th century",
        "associated_minerals": "Silver, manganese, and bauxite",
        "technology_used": "[Underground] and open-pit mining, with increasing use of modern technology"
    },
    "Mexico": {
        "estimated_reserves": "World leader in [silver] production, with [hundreds of millions] of ounces annually",
        "extraction_cost": "[Variable], influenced by ore grade and mining method",
        "grade": "[Very high]-grade silver deposits in regions like Zacatecas",
        "accessibility": "[Excellent], with a long history of mining and developed infrastructure",
        "reserve_life": "[Several decades], with potential for new discoveries",
        "ownership": "[Mixed] ownership with both national and foreign companies active",
        "regulatory_status": "[Evolving], with a focus on environmental and social governance",
        "market_demand": "Consistently [high] for silver, used in industrial applications and investment",
        "historical_data": "[Centuries] of silver mining heritage, integral to national development",
        "associated_minerals": "[Gold, lead, and zinc], commonly found with silver",
        "technology_used": "[Advanced mining] techniques including underground and open-pit mining"
    },
    "Sweden": {
        "estimated_reserves": "[Over 25 million tonnes] of iron ore annually, [significant] copper production",
        "extraction_cost": "[Moderate], due to technologically advanced mining operations",
        "grade": "[High]-grade iron ore, with [significant] copper deposits",
        "accessibility": "[Excellent], with well-developed mining infrastructure and transport",
        "reserve_life": "[Several decades] for iron ore and copper",
        "ownership": "Major companies include [LKAB] for iron ore and [Boliden] for copper",
        "regulatory_status": "[Stringent] environmental regulations with a focus on sustainable mining",
        "market_demand": "[Strong] global demand for high-grade iron ore and copper",
        "historical_data": "[Centuries] of mining history, with a strong tradition in metals",
        "associated_minerals": "[Lead, zinc, and silver]",
        "technology_used": "[Cutting-edge], including autonomous mining technology and electrification"
    },
    "Zambia": {
        "estimated_reserves": "[Second-largest copper producer in Africa], with [millions of tonnes] annually",
        "extraction_cost": "[Medium], influenced by the depth and grade of mines",
        "grade": "Varied, with some of the world's [largest high-grade copper deposits]",
        "accessibility": "[Good], major mining regions well-connected to export routes",
        "reserve_life": "[Extensive], with ongoing exploration expected to extend reserves",
        "ownership": "[Mix] of state-owned entities and international mining corporations",
        "regulatory_status": "[Regulatory environment evolving] with focus on beneficiation",
        "market_demand": "[High], with copper essential for electrical infrastructure and renewable technologies",
        "historical_data": "[Rich] mining history, pivotal to the country's economy",
        "associated_minerals": "[Cobalt, gold, and silver]",
        "technology_used": "[Predominantly open-pit mining], with some underground operations"
    },
    "Democratic Republic of Congo": {
        "estimated_reserves": "[World's leading cobalt producer], critical for rechargeable batteries",
        "extraction_cost": "[Varied], with some challenges due to remote locations",
        "grade": "[Highest-grade cobalt reserves] globally",
        "accessibility": "[Challenging], with mines often located in conflict-prone areas",
        "reserve_life": "[Extensive], with DRC dominating global cobalt reserves",
        "ownership": "[Mix] of state, private, and international mining companies",
        "regulatory_status": "[Complex], with ongoing efforts to regulate the sector and improve transparency",
        "market_demand": "[Surging], driven by the demand for electric vehicles",
        "historical_data": "[Decades] of mining, but recent years have seen exponential growth in cobalt production",
        "associated_minerals": "[Copper, diamonds, and gold]",
        "technology_used": "[Varied], from artisanal to advanced mechanized mining methods"
    },
    "Zimbabwe": {
        "estimated_reserves": "[One of the top platinum producers] globally, [725 millions] significant diamond reserves",
        "extraction_cost": "[High], due to deep level mining for platinum",
        "grade": "[High-grade] platinum group metals, with [significant] diamond deposits",
        "accessibility": "[Moderate], with infrastructure challenges",
        "reserve_life": "[Decades], with potential for expansion through exploration",
        "ownership": "[Mix] of government and international mining companies",
        "regulatory_status": "[Evolving], with efforts to stabilize the mining sector and attract investment",
        "market_demand": "[High] for platinum, used in automotive catalysts and jewelry; [stable] for diamonds",
        "historical_data": "[Rich] in minerals, with recent years seeing growth in platinum and diamond mining",
        "associated_minerals": "[Nickel, palladium, and rhodium]",
        "technology_used": "[Advanced] mining and processing techniques for platinum; [varied] for diamonds"
    },
    "Mongolia": {
        "estimated_reserves": "Significant quantities of coal around [2.5 billion], notable for copper and gold",
        "extraction_cost": "[Varied], generally low for coal due to surface mining",
        "grade": "High-grade coal deposits, [varied] for copper and gold",
        "accessibility": "[Moderate], challenges due to remote locations",
        "reserve_life": "[Extensive] for coal, with substantial potential for copper and gold",
        "ownership": "[Mix] of state and foreign investments",
        "regulatory_status": "[Evolving], with a focus on attracting foreign investment",
        "market_demand": "[High] for coal in neighboring countries, growing for copper and gold",
        "historical_data": "[Decades] of mining history, rapidly growing sector",
        "associated_minerals": "[Silver, molybdenum], and rare earth elements",
        "technology_used": "[Open-pit] and underground mining, depending on the mineral"
    },
    "Saudi Arabia": {
        "estimated_reserves": "Large-scale phosphate production estimated [100 million], emerging gold mining",
        "extraction_cost": "[Low to medium], benefiting from modern extraction technologies",
        "grade": "[High-quality] phosphate rock, [variable] grades for gold",
        "accessibility": "[High], facilitated by significant infrastructure investments",
        "reserve_life": "[Long-term] for phosphate, with ongoing exploration for gold",
        "ownership": "[State-owned] entities, with some foreign joint ventures",
        "regulatory_status": "[Supportive], with efforts to diversify the economy through mining",
        "market_demand": "[Growing], especially for phosphate in agriculture",
        "historical_data": "[Recent expansion] into mining beyond oil",
        "associated_minerals": "Varied, including [copper, zinc], and bauxite",
        "technology_used": "[Advanced], focusing on sustainable and efficient extraction"
    },
    "United Arab Emirates": {
        "estimated_reserves": "Major aluminum producer at around [720 million], leveraging bauxite imports",
        "extraction_cost": "[Competitive], due to efficient smelting operations",
        "grade": "Not applicable for aluminum smelting",
        "accessibility": "[High], strategic location for raw material imports and product exports",
        "reserve_life": "Not applicable, as primary production is from imported bauxite",
        "ownership": "[State-controlled], with significant investments in smelting capacity",
        "regulatory_status": "[Favorable], with a focus on industrial growth",
        "market_demand": "[Strong], with aluminum used across sectors globally",
        "historical_data": "[Rapid growth] in production over the last two decades",
        "associated_minerals": "Not applicable",
        "technology_used": "[State-of-the-art] smelting technology"
    },
    "Turkey": {
        "estimated_reserves": "Leading global producer of boron with estimated [250 million], significant marble exports",
        "extraction_cost": "[Low] for marble, [medium] for boron due to processing",
        "grade": "[High] quality for both marble and boron",
        "accessibility": "[Good], with extensive mining areas and developed infrastructure",
        "reserve_life": "[Extensive] for boron, [abundant] marble reserves",
        "ownership": "[State-dominated] for boron, mixed for marble",
        "regulatory_status": "[Regulatory environment evolving], with incentives for mining sector",
        "market_demand": "[High] for boron in industrial applications, stable for marble",
        "historical_data": "[Centuries] of marble extraction, [decades] of boron dominance",
        "associated_minerals": "[Trona, salt], and other industrial minerals",
        "technology_used": "[Advanced] for boron extraction and processing, traditional to modern for marble"
    },
    "Norway": {
        "estimated_reserves": "[Approx. 2 million barrels] of petroleum per day, significant metals production",
        "extraction_cost": "[High] for petroleum due to offshore platforms, [varied] for metals",
        "grade": "High-quality petroleum, [varied] grades for metals",
        "accessibility": "[High] for offshore oil fields, [moderate] for metal mining areas",
        "reserve_life": "[20+ years] for petroleum, [extensive] for certain metals",
        "ownership": "[State-controlled] (Statoil) for petroleum, mixed for metals",
        "regulatory_status": "[Stringent] environmental and safety standards for offshore drilling",
        "market_demand": "[Stable] for petroleum, [growing] for specific metals like copper",
        "historical_data": "[Decades] of oil export, long history in metal mining",
        "associated_minerals": "[Copper, zinc], and rare earth elements",
        "technology_used": "[Advanced] offshore drilling techniques, modern metal mining methods"
    },
    "Vietnam": {
        "estimated_reserves": "[Approx. 20 million tonnes] of bauxite annually, growing in rare earth elements",
        "extraction_cost": "[Low] to [moderate], depending on the mining area",
        "grade": "High-grade [bauxite], [varied] for rare earth elements",
        "accessibility": "[Moderate], with significant investments in mining infrastructure",
        "reserve_life": "[30+ years] for bauxite, [to be explored] for rare earth elements",
        "ownership": "[State-owned] enterprises primarily, with some foreign investment",
        "regulatory_status": "[Evolving], with focus on sustainable and responsible mining",
        "market_demand": "[High] for bauxite, [increasingly significant] for rare earth elements",
        "historical_data": "[Growing] bauxite production in the past decade, exploration for rare earths",
        "associated_minerals": "Iron ore, [tin], and fluorite",
        "technology_used": "[Surface mining] for bauxite, [varied] for rare earth elements"
    },
    "Nigeria": {
        "estimated_reserves": "[Approx. 2 million barrels] of oil per day, [emerging] in solid minerals",
        "extraction_cost": "[Varied], generally low for onshore oil production",
        "grade": "High-quality light crude oil, [varied] for minerals",
        "accessibility": "[High] for oil fields, [developing] for mineral sites",
        "reserve_life": "[20+ years] for oil, [exploratory] for solid minerals",
        "ownership": "[Mixed] state and foreign ownership in oil, [emerging] private in minerals",
        "regulatory_status": "[Complex], with ongoing reforms to attract investment",
        "market_demand": "[Stable] for oil, [growing] for minerals like gold",
        "historical_data": "[Decades] of oil export, nascent but promising mineral sector",
        "associated_minerals": "[Gold, zinc], and lead",
        "technology_used": "[Offshore and onshore] drilling for oil, traditional mining for minerals"
    },
    "Tanzania": {
        "estimated_reserves": "[Approx. 50 tonnes] of gold annually, significant gemstone production",
        "extraction_cost": "[Moderate], due to varied mining conditions",
        "grade": "High-grade gold deposits, [varied] for gemstones",
        "accessibility": "[Moderate to high], well-developed mining areas for gold and gemstones",
        "reserve_life": "[25+ years] for gold, extensive for gemstones",
        "ownership": "[Mixed] local and foreign investment in mining sector",
        "regulatory_status": "[Favorable], with efforts to attract more mining investment",
        "market_demand": "[High] for gold, particularly from Asia; [stable] for gemstones",
        "historical_data": "[Rich] mining history with significant recent growth in gold production",
        "associated_minerals": "Diamond, [tanzanite], and nickel",
        "technology_used": "[Open-pit and underground mining] for gold, artisanal to semi-industrial for gemstones"
    },
    "Papua New Guinea": {
        "estimated_reserves": "[Approx. 65 tonnes] of gold and [200,000 tonnes] of copper annually",
        "extraction_cost": "[Varied], generally moderate due to remote locations",
        "grade": "[High]-grade gold and copper deposits",
        "accessibility": "[Challenging], due to rugged terrain and limited infrastructure",
        "reserve_life": "[40+ years], with ongoing exploration activities",
        "ownership": "[Mixed] between state and foreign entities, including major mining companies",
        "regulatory_status": "[Regulatory challenges], with efforts to ensure benefits to local communities",
        "market_demand": "[High] globally, especially for copper as an essential industrial metal",
        "historical_data": "[Decades of mining], contributing significantly to the national economy",
        "associated_minerals": "[Silver, nickel], potential for rare earth elements",
        "technology_used": "[Open-pit and underground mining], with advanced processing facilities"
    },
    "Iran": {
        "estimated_reserves": "[Approx. 220,000 tonnes] of zinc, [1 million tonnes] of copper annually",
        "extraction_cost": "[Low to medium], benefiting from large-scale operations",
        "grade": "[High]-quality zinc, varied copper and iron ore grades",
        "accessibility": "[Moderate], with major sites well-connected to infrastructure",
        "reserve_life": "[50+ years] for zinc, extensive for copper and iron ore",
        "ownership": "[State-dominated], with recent openings for foreign investment",
        "regulatory_status": "[Complex], affected by international sanctions and domestic policies",
        "market_demand": "[Growing] for copper and iron ore, stable for zinc",
        "historical_data": "[Ancient mining history], modern development since the 20th century",
        "associated_minerals": "[Lead, silver], and a variety of other minerals",
        "technology_used": "[Modern mining and processing technologies], with ongoing upgrades"
    },
    "Ukraine": {
        "estimated_reserves": "[Approx. 70 million tonnes] of iron ore annually",
        "extraction_cost": "[Low], due to the high-grade nature of the deposits",
        "grade": "[Very high]-grade iron ore, significant coal deposits",
        "accessibility": "[High], with established mining and transportation infrastructure",
        "reserve_life": "[Over 100 years] for iron ore, [declining] for coal",
        "ownership": "[Privately owned and state-owned] enterprises",
        "regulatory_status": "[Evolving], with reforms aimed at attracting investment",
        "market_demand": "[Strong] globally for iron ore, domestic and regional for coal",
        "historical_data": "[Rich industrial heritage], key player in the global steel industry",
        "associated_minerals": "[Manganese, titanium], and rare earth elements",
        "technology_used": "[Open-pit and underground mining], adopting green technologies"
    },
    "Poland": {
        "estimated_reserves": "[Approx. 115 million tonnes] of coal annually",
        "extraction_cost": "[Moderate], influenced by the depth and quality of coal seams",
        "grade": "[Varied], with both thermal and metallurgical coal",
        "accessibility": "[High], well-developed coal mining regions",
        "reserve_life": "[30+ years], with efforts to diversify energy sources",
        "ownership": "[State-controlled], with some private entities",
        "regulatory_status": "[Transitioning], amid EU environmental directives",
        "market_demand": "[Domestic] energy production, [decreasing] export due to global shift",
        "historical_data": "[Centuries of coal mining], an integral part of national identity",
        "associated_minerals": "Minor [metals and industrial minerals], focus remains on coal",
        "technology_used": "[Advanced] coal extraction and processing, moving towards cleaner methods"
    },
    "Bolivia": {
        "estimated_reserves": "[Approx. 1,200 tonnes] of silver annually, emerging lithium production",
        "extraction_cost": "[Moderate] for silver, [high] for lithium due to extraction complexities",
        "grade": "[High]-grade silver deposits, significant lithium brine concentrations",
        "accessibility": "[Challenging], due to remote locations and infrastructure limitations",
        "reserve_life": "[20+ years] for silver, [extensive] for lithium",
        "ownership": "[Mixed] state control and foreign investment",
        "regulatory_status": "[Evolving], with strategic focus on lithium",
        "market_demand": "[High] for both silver and lithium, particularly lithium for battery production",
        "historical_data": "[Long history] of silver mining, lithium production in development",
        "associated_minerals": "[Tin, gold], with potential for rare earth elements",
        "technology_used": "[Traditional mining] for silver, [evolving technologies] for lithium extraction"
    },
    "Namibia": {
        "estimated_reserves": "[Approx. 5,500 tonnes] of uranium annually, significant diamond production",
        "extraction_cost": "[Low] for uranium due to open-pit mining, [varied] for diamonds",
        "grade": "[High]-grade uranium, varied grades for diamonds",
        "accessibility": "[High], with established infrastructure and mining operations",
        "reserve_life": "[40+ years] for uranium, extensive for diamonds",
        "ownership": "[International partnerships], with significant foreign investment",
        "regulatory_status": "[Well-regulated], supportive of mining industry",
        "market_demand": "[Strong] for uranium, stable for diamonds",
        "historical_data": "[Decades] of uranium and diamond mining",
        "associated_minerals": "[Copper, gold], and rare earth elements",
        "technology_used": "[Open-pit mining] for uranium, marine and land-based for diamonds"
    },
    "Botswana": {
        "estimated_reserves": "[Approx. 20 million carats] of diamonds annually",
        "extraction_cost": "[Varied], largely efficient due to scale and technology",
        "grade": "[High]-quality gemstone diamonds",
        "accessibility": "[High], with well-developed mining infrastructure",
        "reserve_life": "[30+ years], with ongoing exploration",
        "ownership": "[Joint venture] between government and De Beers",
        "regulatory_status": "[Stable and supportive], with policies favoring sustainable development",
        "market_demand": "[High], driven by gemstone quality",
        "historical_data": "[Since 1970s], major contributor to GDP",
        "associated_minerals": "Minor [copper, nickel] occurrences",
        "technology_used": "[Advanced] diamond mining and processing techniques"
    },
    "New Zealand": {
        "estimated_reserves": "[Approx. 3 million tonnes] of coal annually, [varied] gold production",
        "extraction_cost": "[Moderate] for both coal and gold, depending on the mining method",
        "grade": "[Varied] coal qualities, high-grade gold in some areas",
        "accessibility": "[Moderate to high], with some remote and environmentally sensitive areas",
        "reserve_life": "[20+ years] for coal, [varied] for gold",
        "ownership": "[Mixed] between government and private entities",
        "regulatory_status": "[Strict environmental oversight], with a focus on sustainable practices",
        "market_demand": "[Domestic and regional] for coal, [high] for gold",
        "historical_data": "[Rich mining history], with a resurgence in gold exploration",
        "associated_minerals": "[Silver, rare earth elements], and industrial minerals",
        "technology_used": "[Open-pit and underground mining], modern gold extraction methods"
    },
    "Finland": {
        "estimated_reserves": "[Approx. 40,000 tonnes] of nickel annually, significant chromium production",
        "extraction_cost": "[Moderate], benefiting from efficient extraction methods",
        "grade": "High-grade [nickel] deposits, significant chromium content",
        "accessibility": "[High], with well-developed mining infrastructure and technology",
        "reserve_life": "[30+ years], with ongoing exploration and development",
        "ownership": "[Private sector] led, with international investments",
        "regulatory_status": "[Pro-mining] policies with a focus on sustainability and environmental protection",
        "market_demand": "[Strong] globally, particularly for nickel in the stainless steel and battery sectors",
        "historical_data": "[Long mining history], with modern technologies enhancing production",
        "associated_minerals": "[Cobalt, copper], and platinum group elements",
        "technology_used": "[Open-pit and underground mining], advanced processing techniques"
    },
    "Mali": {
        "estimated_reserves": "[Approx. 50 tonnes] of gold annually",
        "extraction_cost": "[Low to moderate], due to surface mining operations",
        "grade": "[High]-quality gold deposits",
        "accessibility": "[Moderate], with some areas challenged by remote locations and security issues",
        "reserve_life": "[20+ years], with potential for new discoveries",
        "ownership": "[Joint ventures] between government and international mining companies",
        "regulatory_status": "[Favorable] mining laws, though political stability is a concern",
        "market_demand": "[High] demand for gold on the global market",
        "historical_data": "[Rich gold mining heritage], with significant contributions to the economy",
        "associated_minerals": "Minor [silver] and base metals",
        "technology_used": "[Open-pit mining], with increasing use of modern extraction and processing"
    },
    "Burkina Faso": {
        "estimated_reserves": "[Approx. 40 tonnes] of gold annually",
        "extraction_cost": "[Moderate], influenced by operational and security challenges",
        "grade": "Varied gold grades, [high] in some key mines",
        "accessibility": "[Challenging], due to infrastructural constraints and security concerns",
        "reserve_life": "[25+ years], with extensive ongoing exploration",
        "ownership": "[International mining companies], with government partnerships",
        "regulatory_status": "[Pro-mining] environment, with efforts to improve governance and stability",
        "market_demand": "[Continued high] demand for gold internationally",
        "historical_data": "[Growing mining sector], rapidly becoming a significant gold producer",
        "associated_minerals": "[Zinc, manganese], alongside gold",
        "technology_used": "[Heap leaching], open-pit mining, and some underground operations"
    },
    "Colombia": {
        "estimated_reserves": "[Approx. 85 million tonnes] of coal annually, [world-leading] emerald production",
        "extraction_cost": "[Varied], depending on the mine and mineral",
        "grade": "[High]-quality coal (thermal and metallurgical), top-quality emeralds",
        "accessibility": "[Mixed], with some remote and challenging locations",
        "reserve_life": "[20+ years] for coal, [extensive] for emeralds",
        "ownership": "[Mixed] state and private, including foreign investment for coal",
        "regulatory_status": "[Complex], with efforts to balance mining growth and environmental protection",
        "market_demand": "[Strong] for both coal (regional demand) and emeralds (global luxury market)",
        "historical_data": "[Centuries] of emerald mining, significant coal production growth in recent decades",
        "associated_minerals": "[Nickel, gold], and other precious stones",
        "technology_used": "[Underground mining] for emeralds, [open-pit and underground mining] for coal"
    },
    "Qatar": {
        "estimated_reserves": "[Approx. 77 million tonnes] of LNG annually, significant petroleum production",
        "extraction_cost": "[Low], due to highly efficient extraction methods",
        "grade": "[High]-quality natural gas, with substantial petroleum reserves",
        "accessibility": "[High], with state-of-the-art infrastructure for extraction and export",
        "reserve_life": "[50+ years], among the highest in the world for natural gas",
        "ownership": "[State-controlled], with Qatar Petroleum as the leading entity",
        "regulatory_status": "[Favorable], with a focus on sustainable development and international partnerships",
        "market_demand": "[Very high], particularly for LNG, with strong international export markets",
        "historical_data": "[Decades of development], leading to a world-leading position in LNG production",
        "associated_minerals": "Minor [petroleum] byproducts",
        "technology_used": "[Advanced LNG technology], offshore drilling for petroleum"
    },
    "Egypt": {
        "estimated_reserves": "[Approx. 6 million tonnes] of phosphate annually, growing gold production",
        "extraction_cost": "[Varied], depending on the mineral and extraction site",
        "grade": "[High]-grade phosphate deposits, variable gold quality",
        "accessibility": "[Mixed], with challenges in remote desert locations",
        "reserve_life": "[30+ years] for phosphate, gold exploration ongoing",
        "ownership": "[Government and private sector] partnerships",
        "regulatory_status": "[Evolving], with recent reforms to encourage mining investment",
        "market_demand": "[High] for phosphate, used globally in agriculture; growing for gold",
        "historical_data": "[Long history] of mineral exploration, with recent focus on modernizing the sector",
        "associated_minerals": "[Iron, quartz], alongside main commodities",
        "technology_used": "[Open-pit mining] for phosphate, modern gold processing techniques"
    },
    "Oman": {
        "estimated_reserves": "[Approx. 8 million tonnes] of gypsum annually, emerging copper production",
        "extraction_cost": "[Moderate], benefiting from open-pit mining efficiencies",
        "grade": "[High]-purity gypsum, exploratory copper deposits",
        "accessibility": "[High], with well-developed mining areas and port access",
        "reserve_life": "[20+ years] for gypsum, with potential for copper",
        "ownership": "[Public-private partnerships], with international investment in copper exploration",
        "regulatory_status": "[Proactive], with incentives for mining and sustainable practices",
        "market_demand": "[Strong] for gypsum in the construction industry, interest in copper for electronics",
        "historical_data": "[Decades] of gypsum mining, with recent copper explorations",
        "associated_minerals": "Potential for [gold, chromite], in copper mining areas",
        "technology_used": "[Surface mining] for gypsum, exploratory drilling for copper"
    },
    "Angola": {
        "estimated_reserves": "[Approx. 9 million carats] of diamond annually, significant oil output",
        "extraction_cost": "[Varied], with advancements in diamond extraction",
        "grade": "[High]-grade diamond deposits, quality oil reserves",
        "accessibility": "[Challenging] in some diamond areas, developed oil extraction infrastructure",
        "reserve_life": "[30+ years] for diamonds, extensive oil reserves",
        "ownership": "[Mix of state and foreign investment], with strategic partnerships",
        "regulatory_status": "[Reforming], to attract more investment in both diamonds and oil sectors",
        "market_demand": "[High] globally for diamonds, strong for oil despite shifting energy landscapes",
        "historical_data": "[Rich history] of diamond mining, key player in African oil production",
        "associated_minerals": "Emerging [gold and iron ore] prospects",
        "technology_used": "[Advanced diamond recovery techniques], offshore oil drilling"
    },
    "Kuwait": {
        "estimated_reserves": "[Approx. 2.7 million barrels per day] of petroleum",
        "extraction_cost": "[Low], due to large, easily accessible reserves",
        "grade": "High-quality [crude oil] with low sulfur content",
        "accessibility": "[High], with well-developed infrastructure",
        "reserve_life": "[Over 100 years], based on current reserves and production rates",
        "ownership": "[State-owned], with Kuwait Petroleum Corporation as the main entity",
        "regulatory_status": "[Stable], with a focus on maintaining production levels and sustainability",
        "market_demand": "[High], consistently strong global demand for crude oil",
        "historical_data": "[Decades] of oil production history, key member of OPEC",
        "associated_minerals": "[Gas] is often co-produced with oil",
        "technology_used": "[Advanced drilling] and extraction techniques"
    },
    "Libya": {
        "estimated_reserves": "[Approx. 1.2 million barrels per day] of oil",
        "extraction_cost": "[Varies], influenced by geopolitical stability",
        "grade": "[High]-quality, light sweet crude oil",
        "accessibility": "[Variable], impacted by political and security conditions",
        "reserve_life": "[85+ years], among the highest in Africa",
        "ownership": "[National Oil Corporation] controls the oil sector",
        "regulatory_status": "[Complex], affected by internal conflict and international agreements",
        "market_demand": "[High], particularly for its light sweet crude suitable for refining",
        "historical_data": "[Long-standing] oil production, with peaks and troughs due to instability",
        "associated_minerals": "Minor [natural gas] production",
        "technology_used": "Depends on the field; [ranging from traditional] to more modern methods"
    },
    "Bahrain": {
        "estimated_reserves": "[Approx. 50,000 barrels per day] of oil, significant natural gas",
        "extraction_cost": "[Moderate], due to aging fields and the need for enhanced recovery techniques",
        "grade": "[Varied]; older fields have heavier oil, newer offshore fields have lighter oil",
        "accessibility": "[Mixed]; offshore fields are more challenging",
        "reserve_life": "[30 years] for oil, longer for natural gas",
        "ownership": "[State-owned], with Bahrain Petroleum Company (Bapco) as a key player",
        "regulatory_status": "[Pro-investment], with efforts to attract foreign expertise and capital",
        "market_demand": "[Stable], with regional markets for oil and a growing interest in gas",
        "historical_data": "One of the [earliest producers] in the Middle East, with a history of oil production since the 1930s",
        "associated_minerals": "[Sulfur] and other byproducts from gas processing",
        "technology_used": "[Enhanced oil recovery] techniques for oil, advanced methods for gas extraction"
    },
    "Bangladesh": {
        "estimated_reserves": "[Approx. 20 billion cubic meters] of natural gas annually",
        "extraction_cost": "[Low] to moderate, depending on the field",
        "grade": "[High]-quality natural gas",
        "accessibility": "[Moderate], with both onshore and offshore fields",
        "reserve_life": "[15+ years], with potential for new discoveries",
        "ownership": "[State-owned], with Petrobangla as the leading entity",
        "regulatory_status": "[Supportive], with policies aimed at increasing domestic production and reducing imports",
        "market_demand": "[High], driven by domestic energy needs",
        "historical_data": "[Several decades] of production, with a focus on self-sufficiency",
        "associated_minerals": "Minor [condensate] production associated with [gas] extraction",
        "technology_used": "[Standard drilling] and production techniques for onshore, increasing exploration offshore"
    },
    "Cuba": {
        "estimated_reserves": "[Approx. 70,000 tonnes] of nickel annually",
        "extraction_cost": "[Moderate], due to aging infrastructure",
        "grade": "High-grade [nickel] and cobalt ores",
        "accessibility": "[Moderate], with some mines located in remote areas",
        "reserve_life": "[30+ years], with significant undeveloped deposits",
        "ownership": "[State-controlled], with foreign investment in some projects",
        "regulatory_status": "[Open to foreign investment], but with regulatory challenges",
        "market_demand": "[High], particularly for nickel used in stainless steel and batteries",
        "historical_data": "[Decades] of nickel mining history, one of the largest reserves of cobalt",
        "associated_minerals": "[Cobalt], which is often co-mined with nickel",
        "technology_used": "[Open-pit and underground mining], with a focus on improving efficiency"
    },
    "Venezuela": {
        "estimated_reserves": "[Approx. 2 million barrels per day] of oil",
        "extraction_cost": "[Low], with some of the world's largest oil reserves",
        "grade": "[Varied]; includes both heavy and light crude oil",
        "accessibility": "[Varies], with some reserves in challenging locations like the Orinoco Belt",
        "reserve_life": "[Over 200 years], based on current reserves and extraction rates",
        "ownership": "[State-owned], with PDVSA as the primary operator",
        "regulatory_status": "[Complex], with political and economic challenges affecting the industry",
        "market_demand": "[High], though impacted by global oil prices and sanctions",
        "historical_data": "[Rich oil history], once among the top oil exporters in the world",
        "associated_minerals": "Minor [natural gas] and other hydrocarbons",
        "technology_used": "A mix of [conventional and heavy oil extraction] methods"
    },
    "Suriname": {
        "estimated_reserves": "[Approx. 1.5 million tonnes] of bauxite annually",
        "extraction_cost": "[Low to moderate], benefiting from surface mining",
        "grade": "[High]-quality bauxite, with efficient refining capabilities",
        "accessibility": "[High], with deposits close to the surface",
        "reserve_life": "[20+ years], with potential for new discoveries",
        "ownership": "[State involvement and foreign companies], with plans for sector expansion",
        "regulatory_status": "[Favorable], with government support for mining sector",
        "market_demand": "[Stable], driven by global aluminum production needs",
        "historical_data": "[Long bauxite mining tradition], critical to the country's economy",
        "associated_minerals": "[Minor associated minerals], mainly focused on bauxite",
        "technology_used": "[Surface mining] methods predominant, with modern refining processes"
    },
    "Guinea": {
        "estimated_reserves": "[Approx. 82 million tonnes] of bauxite annually",
        "extraction_cost": "[Low], with some of the largest and highest quality reserves globally",
        "grade": "[Very high]-quality bauxite, suitable for efficient aluminum production",
        "accessibility": "[Good], though infrastructure challenges exist in some areas",
        "reserve_life": "[100+ years], with extensive undeveloped deposits",
        "ownership": "[Mixed ownership], with significant foreign investment",
        "regulatory_status": "[Improving], with efforts to increase transparency and investment",
        "market_demand": "[Very high], as the largest global supplier of bauxite",
        "historical_data": "[Decades of production], increasingly important on the global stage",
        "associated_minerals": "Minor deposits of [iron ore, gold, and diamonds]",
        "technology_used": "[Surface mining] dominates, with ongoing investments in infrastructure"
    },
    "Senegal": {
        "estimated_reserves": "[Approx. 2.5 million tonnes] of phosphate annually",
        "extraction_cost": "[Moderate], due to well-established mining operations",
        "grade": "[High]-grade phosphate rock, suitable for various applications",
        "accessibility": "[High], with phosphate deposits close to the coast for easy export",
        "reserve_life": "[Over 25 years], with potential for further exploration",
        "ownership": "[State and foreign partnerships], aiming to expand production",
        "regulatory_status": "[Favorable] for mining investment, with government support",
        "market_demand": "[Stable], driven by global agricultural needs for fertilizer",
        "historical_data": "[Several decades] of phosphate mining history, integral to the economy",
        "associated_minerals": "Minor occurrences of [gold] and heavy minerals",
        "technology_used": "[Open-pit mining] for phosphate extraction and beneficiation"
    },
    "Cameroon": {
        "estimated_reserves": "[Emerging], with potential for significant iron ore and bauxite production",
        "extraction_cost": "[Variable], depending on the scale of operations and infrastructure development",
        "grade": "[Varied]; high potential for high-grade iron ore and bauxite",
        "accessibility": "[Moderate], with challenges in infrastructure and transport",
        "reserve_life": "[Not clearly estimated], with ongoing exploration and development projects",
        "ownership": "[Mixed], with government involvement and international partnerships",
        "regulatory_status": "[Developing], with efforts to attract investment in the mining sector",
        "market_demand": "[Growing], especially for iron ore in the global steel industry",
        "historical_data": "[Limited], with recent interest in exploiting mineral resources",
        "associated_minerals": "Potential for [nickel, cobalt, and gold]",
        "technology_used": "[Under development], with potential for modern mining and processing methods"
    },
    "Sierra Leone": {
        "estimated_reserves": "[Approx. 500,000 carats] of diamond annually",
        "extraction_cost": "[High], due to artisanal mining practices and recovery methods",
        "grade": "[High]-quality diamonds, with a reputation for large, gem-quality stones",
        "accessibility": "[Moderate to high], with alluvial diamond fields and kimberlite deposits",
        "reserve_life": "[20+ years], with significant unexplored territories",
        "ownership": "[Mixed], with artisanal miners and formal mining companies",
        "regulatory_status": "[Improving], with efforts to regulate the diamond trade and increase transparency",
        "market_demand": "[High], with a global demand for diamonds in jewelry and industrial applications",
        "historical_data": "[Decades of diamond mining], with a complex history including conflict diamonds",
        "associated_minerals": "[Gold, rutile, and bauxite], with potential for diversified mining operations",
        "technology_used": "[Artisanal and small-scale mining], with some mechanized operations"
    },
    "Cote d'Ivoire": {
        "estimated_reserves": "[Approx. 25,000 tonnes] of gold annually, [emerging manganese production]",
        "extraction_cost": "[Low to moderate], with modern mining techniques improving efficiency",
        "grade": "[High]-grade gold deposits, manganese grade varies by deposit",
        "accessibility": "[High], with mining areas relatively accessible and infrastructure improving",
        "reserve_life": "[30+ years] for gold, manganese reserves less well defined",
        "ownership": "[Mixed], with a strong presence of international mining companies",
        "regulatory_status": "[Pro-mining] policies, with the government encouraging foreign investment",
        "market_demand": "[Strong] for gold on the global market, manganese demand growing with steel production",
        "historical_data": "[Rapid growth] in gold mining over the past decade, manganese exploration recent",
        "associated_minerals": "[Nickel, cobalt, and copper] potential in diversified mineral portfolio",
        "technology_used": "[Modern open-pit and underground mining] for gold, manganese extraction methods developing"
    },
    "Liberia": {
        "estimated_reserves": "[Approx. 5 million tonnes] of iron ore annually",
        "extraction_cost": "[Moderate], due to the need for infrastructure development",
        "grade": "High-grade [iron ore], attractive to global steel manufacturers",
        "accessibility": "[Challenging], given Liberia's infrastructure and logistical constraints",
        "reserve_life": "[20+ years], with significant exploration potential",
        "ownership": "[Mixed], with government and international mining companies",
        "regulatory_status": "[Supportive], with efforts to attract mining investments",
        "market_demand": "[Strong], driven by global steel production",
        "historical_data": "[Rich mining history], with iron ore mining dating back to the 1950s",
        "associated_minerals": "[Gold and diamonds], with potential for exploration",
        "technology_used": "[Open-pit mining] techniques predominantly"
    },
    "Mozambique": {
        "estimated_reserves": "[Approx. 10 million tonnes] of coal annually, [emerging titanium production]",
        "extraction_cost": "[Variable], depending on the mine and mineral",
        "grade": "High-quality coal, [significant titanium deposits] in sands",
        "accessibility": "[Moderate], with port and rail infrastructure improving",
        "reserve_life": "[30+ years] for coal, titanium reserves less well defined",
        "ownership": "[International and local partnerships], including major mining companies",
        "regulatory_status": "[Proactive], with the government promoting the mining sector",
        "market_demand": "[High for coal], particularly from India and China; growing for titanium",
        "historical_data": "[Rapid development] of the coal sector in the 21st century",
        "associated_minerals": "[Graphite, ilmenite], part of the country's broad mineral portfolio",
        "technology_used": "[Open-pit mining for coal], dredge mining for titanium"
    },
    "Madagascar": {
        "estimated_reserves": "[Approx. 50,000 tonnes of graphite annually], [emerging nickel production]",
        "extraction_cost": "[Low to moderate], benefiting from surface mining methods",
        "grade": "[High]-grade graphite, world-class nickel deposits",
        "accessibility": "[Varied], with some remote areas presenting challenges",
        "reserve_life": "[25+ years] for graphite, nickel reserves expected to last decades",
        "ownership": "[Foreign and local partnerships], with significant international investment",
        "regulatory_status": "[Evolving], with a focus on sustainable mining practices",
        "market_demand": "[Strong for graphite], driven by battery and tech industries; nickel in steady demand",
        "historical_data": "[Long history] of graphite mining; nickel operations more recent",
        "associated_minerals": "[Cobalt, rare earth elements], enhancing Madagascar's mining portfolio",
        "technology_used": "[Open-pit mining] for graphite, lateritic nickel mining"
    },
    "Lesotho": {
        "estimated_reserves": "[Approx. 1 million carats] of diamond annually",
        "extraction_cost": "[High], due to the need for sophisticated diamond recovery technologies",
        "grade": "[Very high]-value diamonds, including some of the world's largest and most valuable gems",
        "accessibility": "[Moderate], with diamond mines located in challenging terrains",
        "reserve_life": "[20+ years], with ongoing exploration likely to extend this",
        "ownership": "[Joint ventures] between the Lesotho government and international companies",
        "regulatory_status": "[Stable], with a regulatory environment conducive to diamond mining",
        "market_demand": "[High], particularly for large and high-quality gemstones",
        "historical_data": "[Notable discoveries], including some of the largest diamonds in recent decades",
        "associated_minerals": "[None] significant, with the focus almost exclusively on diamonds",
        "technology_used": "[Underground mining], employing advanced diamond recovery techniques"
    },
    "Ethiopia": {
        "estimated_reserves": "[Approx. 5,000 tonnes] of opal annually, [emerging gold and tantalum production]",
        "extraction_cost": "[Low to moderate], with artisanal mining methods prevalent",
        "grade": "[High]-quality opals, gold and tantalum grades vary",
        "accessibility": "[Moderate], with infrastructure development being a focus",
        "reserve_life": "[20+ years], with significant untapped potential",
        "ownership": "[Mixed], government and private entities, including international partnerships",
        "regulatory_status": "[Developing], with efforts to modernize mining laws",
        "market_demand": "[High for opals], growing for gold and tantalum due to tech applications",
        "historical_data": "[Rich in tradition], with a long history of opal mining",
        "associated_minerals": "[Copper, limestone], with potential for further exploration",
        "technology_used": "[Artisanal mining methods], transitioning to more modern techniques"
    },
    "Kyrgyzstan": {
        "estimated_reserves": "[Approx. 20 tonnes] of gold annually, significant coal and antimony",
        "extraction_cost": "[Moderate], due to mountainous terrain and remote locations",
        "grade": "[High-grade gold], with coal and antimony varying in quality",
        "accessibility": "[Challenging], given Kyrgyzstan's topography",
        "reserve_life": "[30+ years], with extensive exploration ongoing",
        "ownership": "[Mixed ownership], including significant foreign investment in gold mining",
        "regulatory_status": "[Complex], with mining regulations undergoing reforms",
        "market_demand": "[Strong for gold], steady for coal and antimony",
        "historical_data": "[Historically significant], with a legacy of mining dating back to the Soviet era",
        "associated_minerals": "[Mercury, uranium], with exploration for rare earth elements",
        "technology_used": "[Underground and open-pit mining], with modern processing techniques for gold"
    },
    "Tajikistan": {
        "estimated_reserves": "[Approx. 5 tonnes] of gold and [2,000 tonnes] of silver annually",
        "extraction_cost": "[Varies], with some mines in remote, high-altitude areas",
        "grade": "[Variable grades], with some high-grade silver and gold deposits",
        "accessibility": "[Moderately challenging], due to rugged terrain",
        "reserve_life": "[20+ years], with potential for discovery of new deposits",
        "ownership": "[State and foreign partnerships], focusing on mineral resource development",
        "regulatory_status": "[Evolving], with efforts to attract more foreign investment",
        "market_demand": "[High for both silver and gold], driven by jewelry and industrial uses",
        "historical_data": "[Rich mining heritage], with ancient mining activities",
        "associated_minerals": "[Lead, zinc, antimony], further diversifying the mineral portfolio",
        "technology_used": "[Both traditional and modern mining techniques], adapting to the terrain"
    },
    "Myanmar": {
        "estimated_reserves": "[World-leading in jade], significant production of gems and tin",
        "extraction_cost": "[High for jade], due to complex extraction methods",
        "grade": "[Top quality jade and gemstones], with significant tin deposits",
        "accessibility": "[Challenging], especially for jade mines in the north",
        "reserve_life": "[Decades for jade and gems], with ongoing exploration",
        "ownership": "[Mixed], with government involvement and private entities",
        "regulatory_status": "[Complex regulatory environment], with recent reforms",
        "market_demand": "[Exceptionally high for jade], especially from China; steady for gems and tin",
        "historical_data": "[Centuries-old jade mining tradition], with a long history of gemstone mining",
        "associated_minerals": "[Limestone, copper], among others",
        "technology_used": "[Varied], from artisanal methods for gems to more industrial for tin"
    },
    "Laos": {
        "estimated_reserves": "[Approx. 100,000 tonnes] of potash annually, emerging gold production",
        "extraction_cost": "[Moderate], due to developing infrastructure",
        "grade": "[High-quality potash], gold grade varies",
        "accessibility": "[Improving], with investments in mining infrastructure",
        "reserve_life": "[30+ years], with significant exploration potential",
        "ownership": "[Joint ventures], including international mining companies",
        "regulatory_status": "[Progressive], with incentives for foreign investment in mining",
        "market_demand": "[High for potash], given global agricultural needs; gold always in demand",
        "historical_data": "[Growing industry], with recent discoveries boosting mining sector",
        "associated_minerals": "[Copper, bauxite], with potential for more discoveries",
        "technology_used": "[Conventional mining for potash], open-pit and underground for gold"
    },
    "Brunei": {
        "estimated_reserves": "[Approx. 100,000 barrels] of petroleum per day, significant natural gas",
        "extraction_cost": "[Low], due to accessible reserves and established infrastructure",
        "grade": "[High-quality petroleum], rich natural gas fields",
        "accessibility": "[High], with extensive oil and gas infrastructure",
        "reserve_life": "[20+ years], with ongoing exploration to extend",
        "ownership": "[State-controlled], with strict regulation of the energy sector",
        "regulatory_status": "[Stable], with a focus on sustainable energy production",
        "market_demand": "[High], particularly in Asia for both petroleum and natural gas",
        "historical_data": "[Decades of production], a key player in Southeast Asia's energy market",
        "associated_minerals": "[None significant], with a focus on hydrocarbons",
        "technology_used": "[Advanced], with significant investment in extraction and processing technology"
    },
    "Turkmenistan": {
        "estimated_reserves": "[Approx. 70 billion cubic meters] of natural gas annually",
        "extraction_cost": "[Low to moderate], with large-scale gas fields",
        "grade": "[High]-quality natural gas, with significant reserves",
        "accessibility": "[Varied], with major fields both onshore and offshore",
        "reserve_life": "[50+ years], one of the world’s largest natural gas reserves",
        "ownership": "[State-owned], with a monopoly on gas production",
        "regulatory_status": "[Controlled], with efforts to diversify gas export routes",
        "market_demand": "[High], especially from China and Russia",
        "historical_data": "[Significant producer], with a long history of gas export",
        "associated_minerals": "[Sulfur, salt], byproducts of gas processing",
        "technology_used": "[Modern], with investments in gas extraction and processing"
    },
    "Uzbekistan": {
        "estimated_reserves": "[Approx. 90 tonnes] of gold annually, significant uranium and copper production",
        "extraction_cost": "[Varies], with some remote and challenging mining conditions",
        "grade": "[High]-grade gold, with uranium and copper varying in quality",
        "accessibility": "[Moderate], with some mines in remote areas but good infrastructure",
        "reserve_life": "[20+ years], with ongoing exploration likely to extend",
        "ownership": "[State-controlled] and [foreign joint ventures], with strategic mineral resources",
        "regulatory_status": "[Reforming], with recent efforts to attract foreign mining investment",
        "market_demand": "[High for all minerals], given their use in various industries",
        "historical_data": "[Rich in resources], with a long history of mining",
        "associated_minerals": "[Silver, lead, zinc], often found in conjunction with copper and gold",
        "technology_used": "[Varied], from traditional to high-tech, particularly in gold and uranium extraction"
    }
}


POSSIBLE_DISEASES = [
    "Hypertension",
    "Diabetes Mellitus",
    "Common Cold",
    "Influenza",
    "Asthma",
    "Depression",
    "Anxiety Disorder",
    "Obesity",
    "Allergic Rhinitis",
    "Reflux Esophagitis",
    "Hyperlipidemia",
    "Osteoarthritis",
    "Thyroid Disorder",
    "Chronic Sinusitis",
    "Menopause",
    "Urinary Tract Infection",
    "Eczema",
    "Psoriasis",
    "Celiac Disease",
    "Migraine",
    "Chronic Kidney Disease",
    "Heart Failure",
    "Coronary Artery Disease",
    "Arrhythmia",
    "Pneumonia",
    "Tuberculosis",
    "Hepatitis",
    "Peptic Ulcer Disease",
    "Pancreatitis",
    "Gallstones",
    "Gastroenteritis",
    "Bronchitis",
    "Otitis Media",
    "Sinusitis",
    "Anemia",
    "Chicken Pox",
    "Dengue Fever",
    "Rheumatoid Arthritis",
    "Systemic Lupus Erythematosus",
    "Scoliosis",
    "Sciatica",
    "Herniated Disc",
    "Fibromyalgia",
    "Carpal Tunnel Syndrome",
    "Acne",
    "Rosacea",
    "Melanoma",
    "Lyme Disease",
    "Herpes Zoster",
    "Human Papillomavirus Infection",
    "Chlamydia Infection",
    "Gonorrhea",
    "Syphilis",
    "HIV/AIDS",
    "Candidiasis",
    "Athlete's Foot",
    "Tinea Versicolor",
    "Alopecia Areata",
    "Bipolar Disorder",
    "Schizophrenia",
    "Dementia",
    "Parkinson's Disease",
    "Alzheimer's Disease",
    "Stroke",
    "Multiple Sclerosis",
    "Amyotrophic Lateral Sclerosis",
    "Epilepsy",
    "Meningitis",
    "Bell's Palsy",
    "Conjunctivitis",
    "Cataracts",
    "Glaucoma",
    "Macular Degeneration",
    "Diabetic Retinopathy",
    "Dry Eye Syndrome",
    "Otitis Externa",
    "Tinnitus",
    "Vertigo",
    "Laryngitis",
    "Tonsillitis",
    "Oral Candidiasis",
    "Gingivitis",
    "Periodontitis",
    "Dyspepsia",
    "Irritable Bowel Syndrome",
    "Inflammatory Bowel Disease",
    "Hemorrhoids",
    "Hepatitis C",
    "Liver Cirrhosis",
    "Chronic Obstructive Pulmonary Disease",
    "Pulmonary Embolism",
    "Pulmonary Hypertension",
    "Sarcoidosis",
    "Endometriosis",
    "Polycystic Ovary Syndrome",
    "Erectile Dysfunction",
    "Prostate Enlargement",
    "Kidney Stones",
    "Bladder Infection"
]









