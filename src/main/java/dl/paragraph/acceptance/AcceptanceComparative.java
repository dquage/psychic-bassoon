package dl.paragraph.acceptance;

import dl.paragraph.pojo.Resultat;

/**
 * Accepte selon la précision de tous les éléments.
 * Soit la précision du meilleur résultat dépasse une limite acceptable.
 * Soit la précision du meilleur résultat dépasse un minimum acceptable mais les autres résultats ne dépasse pas un palier.
 */
public class AcceptanceComparative implements Acceptance {

    private int precisionAccepte = 80;
    private int precisionMin = 60;
    private int palier = 40;

    public AcceptanceComparative() {
    }

    public AcceptanceComparative(int precisionAccepte, int precisionMin, int palier) {
        this.precisionAccepte = precisionAccepte;
        this.precisionMin = precisionMin;
        this.palier = palier;
    }

    @Override
    public String accept(Resultat resultat) {

        double score1 = resultat.getCategorie1().getScore() * 100;
        double score2 = resultat.getCategorie2().getScore() * 100;
        double score3 = resultat.getCategorie3().getScore() * 100;

        boolean accepte = (score1 > precisionAccepte)
                || (score1 > precisionMin && score2 < palier && score3 < palier);

        return accepte ? resultat.getCategorie1().getLabel() : null;
    }
}
