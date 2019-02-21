package dl.paragraph.pojo;

import lombok.Builder;
import lombok.Data;

/**
 * Classe pour stocker les résultats sous forme simplifié.
 */
@Data
@Builder
public class Resultat {

    private String libelle;
    private String labelExpected;
    private Categorie categorie1;
    private Categorie categorie2;
    private Categorie categorie3;
    private String labelAccepted;

    public boolean isAccurate() {
        return labelExpected != null && categorie1 != null && labelExpected.equals(categorie1.getLabel());
    }

    public boolean isAccurateLabelAccepted() {
        return labelExpected != null && labelAccepted != null && labelExpected.equals(labelAccepted);
    }
}
