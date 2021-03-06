package dl.paragraph.pojo;

import com.fasterxml.jackson.annotation.JsonValue;
import com.google.common.collect.Lists;

import java.util.List;

public enum Apollon {

    // Dépend du véhicule
    ESSENCE_VEHICULE("ESSENCE VEHICULE", null),
    CARBURANT_VEHICULE("CARBURANT VEHICULE", null),
    DEPENSES_VEHICULE("DEPENSES VEHICULE", null),
    ASSURANCE_VEHICULE("ASSURANCE VEHICULE", null),
    LLD_VEHICULE("LLD VEHICULE", null),
    LEASING_VEHICULE("LEASING VEHICULE", null),
    // Dépend de l'emprunt
    EMPRUNT("EMPRUNT", null), // 164

    // en SCP dépend de l'associé
    COTISATION_CARPIMKO("COTISATION CARPIMKO", "6461000"), // IND 6461000
    COTISATIONS_RETRAITE_OBLIGATOIRE("COTISATIONS RETRAITE OBLIGATOIRE", "6461000"),
    COTISATIONS_URSSAF_DE_L_EXPLOITANT("COTISATIONS URSSAF DE L EXPLOITANT", "6462000"),
    COTISATIONS_MALADIE_EXPLOITANT("COTISATIONS MALADIE EXPLOITANT", "6462010"),
    COTISATIONS_COMPLEMENTAIRES_RETRAITE("COTISATIONS COMPLEMENTAIRES RETRAITE", "6464010"),
    COTISATIONS_COMPLEMENTAIRES_SANTE("COTISATIONS COMPLEMENTAIRES SANTE", "6464020"),
    COTISATIONS_PROF_ET_SYNDICALE("COTISATIONS PROF.ET SYNDICALE", "6280000"),
    COTISATIONS_COMPLEMENTAIRES_EXPLOITANT("COTISATIONS COMPLEMENTAIRES EXPLOITANT", "6464000"),
    PERTE_EMPLOI_MADELIN("PERTE EMPLOI MADELIN", "6464030"),

    COMPTE_ATTENTE("COMPTE D ATTENTE", "4710000"),
    COMPTE_ATTENTE_IMMOBILISATIONS("COMPTE DATTENTE DES IMMOBILISATIONS", "4720000"),
    COMPTE_DE_L_EXPLOITANT("COMPTE DE L EXPLOITANT", "1080000"),
    APPORT_PERSONNEL("APPORT PERSONNEL", "1080100"),
    COMPTE_ASSOCIE("COMPTE ASSOCIE", "4550000"),

    CAPITAL("CAPITAL", "1013000"),
    REINTEGRATION_COMPTABLE("REINTEGRATION COMPTABLE", "1060000"),
    PART_PRIVEE_DES_DEPENSES_MIXTES("PART PRIVEE DES DEPENSES MIXTES", "1070000"),

    DIVERS_A_REINTEGRER("DIVERS A REINTEGRER", "1090100"), // 1090000 ?
    DIVERS_A_DEDUIRE("DIVERS A DEDUIRE", "1090200"),
    RESULTAT_EXERCICE("RESULTAT DE L EXERCICE", "1200000"),


    ACHATS_A_USAGE_UNIQUE("ACHATS USAGE UNIQUE", "6022000"),
    PETIT_OUTILLAGE_PETIT_MATERIEL("PETIT OUTILLAGE PETIT MATERIEL", "6063000"),
    FOURNITURES_DE_BUREAU_ET_ADMINISTRATIVES("FOURNITURES DE BUREAU ET ADMINISTRATIVES", "6064000"),
    ENTRETIEN_ET_REPARATIONS("ENTRETIEN ET REPARATIONS", "6150000"),
    ASSURANCES("ASSURANCES", "6160000"),
    CADEAUX_CLIENTELE("CADEAUX CLIENTELE", "6234000"),
    TRANSPORTS_DIVERS_BUS_TAXIS("TRANSPORTS DIVERS (bus, taxis)", "6248000"),
    DEPLACEMENTS_MISSIONS_RECEPTIONS("DEPLACEMENTS, MISSIONS, RECEPTIONS", "6250000"),
    REPAS_DE_MIDI("REPAS DE MIDI", "6250001"),
    FRAIS_POSTAUX("FRAIS POSTAUX", "6261000"),
    FRAIS_DE_TELECOMMUNICATION("FRAIS DE TELECOMMUNICATION", "6262000"),
    SERVICES_BANCAIRES("SERVICES BANCAIRES", "6270000"),
    AGIOS_BANCAIRES("AGIOS BANCAIRES", "6270100"),
    CFE_CVAE("CFE,CVAE", "6351100"),
    DROIT_ENREGISTREMENT("DROIT D ENREGISTREMENT", "6354000"),
    AUTRES_IMPOTS("AUTRES IMPOTS", "6370000"),
    CSG_DEDUCTIBLE("CSG DEDUCTIBLE", "6371000"),

    REMUNERATIONS_DU_PERSONNEL("REMUNERATIONS DU PERSONNEL", "6410000"),
    SECURITE_SOC_ET_PREVOYANCE_DU_PERSONNEL("SECURITE SOC. ET PREVOYANCE DU PERSONNEL", "6450000"),
    COTISATION_RETRAITE_DU_PERSONNEL("COTISATION RETRAITE DU PERSONNEL", "6453000"),
    COTISATION_ASSEDIC_DU_PERSONNEL("COTISATION ASSEDIC DU PERSONNEL", "6454000"),

    CHARGES_DIVERSES_DE_GESTION_COURANTE("CHARGES DIVERSES DE GESTION COURANTE", "6500000"),
    CHEQUES_VACANCES("CHEQUES VACANCES", "6503000"),
    FORFAIT_BLANCHISSAGE("FORFAIT BLANCHISSAGE", "6509000"),
    CHARGES_FINANCIERES("CHARGES FINANCIERES", "6610000"),
    VALEURS_COMPTABLES_ELEMENTS_CEDES("VALEURS COMPTABLES DES ELEMENTS CEDES", "6750000"),
    DOTATIONS_AMORT_PROVISIONS("DOTATIONS AUX AMORT. ET PROVISIONS", "6800000"),
    DEFICIT_SCM_DOTATIONS("DEFICIT SCM DOTATIONS", "6801000"),


    HONORAIRES("HONORAIRES", "7060000"),
    HONORAIRES_NON_CONVENTIONNES("HONORAIRES NON CONVENTIONNES", "7080000"),
    HONORAIRES_RETROCEDES("HONORAIRES RETROCEDES", "6223000"),
    HONORAIRES_RETROCEDES_A_REMPLACANT("HONORAIRES RETROCEDES A REMPLACANT", "7090000"),

    REDEVANCE_COLLABORATION("REDEVANCE COLLABORATION", "7580200"),
    REDEVANCE_DE_COLLABORATION("REDEVANCE DE COLLABORATION", "7580200"),

    TROP_PERCU("TROP PERCU", "7069999"),
    AIDE_TELETRANSMISSION("AIDE TELETRANSMISSION", "7401000"),
    ACTIVITES_ANNEXES("ACTIVITES ANNEXES", null), // Dépends des activités annexes
    GAINS_DIVERS("GAINS DIVERS", "7580000"),
    INDEMNITE_MATERNITE("INDEMNITE MATERNITE", "7580500"),
    BONUS_ECOLOGIQUE("BONUS ECOLOGIQUE", "7580600"),
    PRODUITS_FINANCIERS("PRODUITS FINANCIERS", "7600000"),
    INDEMNITE_MALADIE("INDEMNITE MALADIE", "7910100"),
    FORMATION("FORMATION", "7910200"),


    FRAIS_DE_FORMATION_SEMINAIRE("FRAIS DE FORMATION SEMINAIRE", "6185000"),
    FRAIS_DE_DOCUMENTATION_GENERALE("FRAIS DE DOCUMENTATION GENERALE", "6181000"),
    FRAIS_DE_DOCUMENTATION_TECHNIQUE("FRAIS DE DOCUMENTATION TECHNIQUE", "6183000"),
    LOCATIONS_IMMOBILIERES("LOCATIONS IMMOBILIERES", "6132000"),
    AUTRES_SERVICES_EXTERIEURS_HONORAIRES("AUTRES SERVICES EXTERIEURS : HONORAIRES", "6226000"),
    PRESTATIONS_DE_SERVICE("PRESTATIONS DE SERVICE", "6040000"),
    LEASING_AUTRE_QUE_VEHICULE("LEASING AUTRE QUE VEHICULE", "6122010"),
    EDF_GDF_CHAUFFAGE("E.D.F., G.D.F., CHAUFFAGE", "6061000"),
    PRODUITS_D_ENTRETIEN("PRODUITS D ENTRETIEN", "6022010"),
    FRAIS_DE_CONGRES("FRAIS DE CONGRES", "6285000"),
    FRAIS_VOITURE_REEL("FRAIS DE VOITURE (REEL)", "6241000"),
    LOCATION_VEHICULE("LOCATION VEHICULE", "6135000"),
    LOCATION_VEHICULE_BIC("LOCATION VEHICULE BIC", "6250200"),
    LOCATION_LD_AUTRE_QUE_VEHICULE("LOCATION LD AUTRE QUE VEHICULE", "6135010"), // LOCATION LD AUTRE QUE VÉHICULE
    FRAIS_D_ACTES_ET_DE_CONTENTIEUX("FRAIS D ACTES ET DE CONTENTIEUX", "6227000"),

    PRODUIT_CESSION_IMMOBILISATION("PRODUIT DE CESSION D IMMOBILISATION", "7750000"),

    IMMOBILISATIONS_INCORPORELLES("IMMOBILISATIONS INCORPORELLES", "2000000"),
    IMMOBILISATIONS_LOGICIELS("IMMOBILISATIONS LOGICIELS", "2050000"),
    IMMOBILISATIONS_CLIENTELE("IMMOBILISATIONS CLIENTELE", "2070000"),
    CONSTRUCTION("CONSTRUCTION", "2130000"),
    IMMO_AGENCEMENTS_AMENAGEMENTS_DIVERS("IMMO. AGENCEMENTS, AMENAGEMENTS DIVERS", "2181000"),
    IMMOBILISATIONS_MATERIELS_TRANSPORT("IMMOBILISATIONS MATERIELS DE TRANSPORT", "2182000"),
    IMMO_MATERIELS_BUREAU_INFORMATIQUE("IMMO. MATERIELS DE BUREAU ET INFORMATIQUE", "2183000"), // <------------ IMMO.MATERIELS DE BUREAU ET INFORMATIQUE
    IMMOBILISATIONS_MOBILIER("IMMOBILISATIONS DU MOBILIER", "2184000"),
    TITRE_PARTICIPATION("TITRE DE PARTICIPATION", "2610000"),
    DEPOTS_CAUTIONNEMENTS_VERSES("DEPOTS ET CAUTIONNEMENTS VERSES", "2750000"),

    AMORTISSEMENTS_IMMOBI_INCORPORELLES("AMORTISSEMENTS DES IMMOBI. INCORPORELLES", "2800000"),
    AMORTISSEMENTS_LOGICIELS("AMORTISSEMENTS LOGICIELS", "2805000"),
    AMORTISSEMENTS_CONSTRUCTIONS("AMORTISSEMENTS CONSTRUCTIONS", "2813000"),
    AMORT_AGENCEMENTS_AMENAGEMENTS_DIVERS("AMORT. AGENCEMENTS, AMENAGEMENTS DIVERS", "2818100"),
    AMORT_IMMO_MATERIELS_TRANSPORT("AMORTI. DES IMMO. MATERIELS DE TRANSPORT", "2818200"),
    AMORT_MATERIELS_BUREAU_INFORMATIQE("AMORT. MATERIELS DE BUREAU & INFORMATIQ.", "2818300"),
    AMORTISSEMENTS_IMMO_MOBILIER("AMORTISSEMENTS DES IMMOB. DU MOBILIER", "2818400"),


    ACHATS_MARCHANDISES("ACHATS MARCHANDISES", "6070000"),

//    CREDIT_BAIL_LEASING_MOBILIER("CREDIT BAIL (LEASING) MOBILIER", "6122010"), // same LEASING AUTRE QUE VEHICULE
    CREDIT_BAIL_LEASING_VEHICULE("CREDIT BAIL (LEASING) VEHICULE", "6122000"),



    CONTRE_PARTIE_AU_COMPTE_109("CONTRE PARTIE AU COMPTE 109", "1080800"),
    CONTRE_PARTIE_AUX_FORFAITS("CONTRE-PARTIE AUX FORFAITS", "1080900"),

    TAXE_VALEUR_AJOUTEE("TAXE SUR LA VALEUR AJOUTEE", "4450000"),
    PERSONNEL_INTERIMAIRE("PERSONNEL INTERIMAIRE", "6211000"),
    FORFAIT_KM("FORFAIT KM", "6249000"),
    TAXE_SALAIRE("TAXE SUR SALAIRE", "6310000");


    private String label;
    private String numero; // Numéro de compte exact

    Apollon(String label, String numero) {
        this.label = label;
        this.numero = numero;
    }

    public static Apollon fromLabel(String label) {
        if (label == null) {
            return null;
        }
        for (Apollon o : Apollon.values()) {
            if (label.equals(o.getLabel())) {
                return o;
            }
        }
        return null;
    }

    public static Apollon fromNumero(String numero) {
        if (numero == null) {
            return null;
        }
        for (Apollon o : Apollon.values()) {
            if (numero.equals(o.getNumero())) {
                return o;
            }
        }
        return null;
    }

    /**
     * Liste de tous les labels.
     * @return
     */
    public static List<String> labels() {
        List<String> labels = Lists.newArrayList();
        for (Apollon o : Apollon.values()) {
            labels.add(o.label);
        }
        return labels;
    }

    @JsonValue
    public String getLabel() {
        return label;
    }

    public String getNumero() {
        return numero;
    }

    @Override
    public String toString() {
        return label;
    }
}
