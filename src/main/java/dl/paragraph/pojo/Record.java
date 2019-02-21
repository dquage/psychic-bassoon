package dl.paragraph.pojo;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class Record {

    private String categorie;
    private String libelle;
    private double montant;
}
