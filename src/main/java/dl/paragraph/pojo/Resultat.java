package dl.paragraph.pojo;

import lombok.Builder;
import lombok.Data;

/**
 * Classe pour stocker les résultats sous forme simplifié.
 */
@Data
@Builder
public class Resultat {

    private Record record;
    private String libelle;
    private Categorie categorie1;
    private Categorie categorie2;
    private Categorie categorie3;
    private String labelAccepted;

    public String getLabelExpected() {
        return record != null ? record.getCategorie() : null;
    }

    public boolean isAccurate() {
        return getLabelExpected() != null && categorie1 != null && getLabelExpected().equals(categorie1.getLabel());
    }

    public boolean isAccurateLabelAccepted() {
        return getLabelExpected() != null && labelAccepted != null && getLabelExpected().equals(labelAccepted);
    }
}
