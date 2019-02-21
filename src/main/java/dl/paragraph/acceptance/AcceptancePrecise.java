package dl.paragraph.acceptance;

import dl.paragraph.pojo.Resultat;

/**
 * Accepte le résultat selon la précision.
 */
public class AcceptancePrecise implements Acceptance {

    private int precision = 60;

    public AcceptancePrecise(int precision) {
        this.precision = precision;
    }

    @Override
    public String accept(Resultat resultat) {
        double score100 = resultat.getCategorie1().getScore() * 100;
        return score100 > precision ? resultat.getCategorie1().getLabel() : null;
    }
}
