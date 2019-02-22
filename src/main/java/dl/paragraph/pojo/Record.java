package dl.paragraph.pojo;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Record {

    private String categorie;
    private String libelle;
    private double montant;

    public boolean isRecette() {
        return montant >= 0;
    }

    public boolean isDepense() {
        return !isRecette();
    }
}
