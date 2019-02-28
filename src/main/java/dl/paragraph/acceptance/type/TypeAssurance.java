package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Apollon;
import dl.paragraph.pojo.Resultat;

public class TypeAssurance implements Acceptance {

    public static final String CATEG_1 = Apollon.ASSURANCES.getLabel();
    public static final String CATEG_2 = Apollon.ASSURANCE_VEHICULE.getLabel();

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
