package dl.paragraph.acceptance.type;

import dl.paragraph.acceptance.Acceptance;
import dl.paragraph.pojo.Resultat;

public class TypeTransports implements Acceptance {

    public static final String CATEG_1 = "TRANSPORTS_DIVERS_BUS_TAXIS";

    @Override
    public String accept(Resultat resultat) {
        return CATEG_1;
    }
}
