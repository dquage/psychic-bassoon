package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeAssurance implements Acceptance {

    public static final String CATEG_1 = "ASSURANCES";
    public static final String CATEG_2 = "ASSURANCE_VEHICULE";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
