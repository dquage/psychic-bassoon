package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeCba implements Acceptance {

    public static final String CATEG_1 = "ENTRETIEN_ET_REPARATIONS";
    public static final String CATEG_2 = "LOCATION_LD_AUTRE_QUE_VEHICULE";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
